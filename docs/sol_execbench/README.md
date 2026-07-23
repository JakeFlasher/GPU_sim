# SOL-ExecBench Counter-Free Profiling Framework

> **RETIRED SUBTREE — HISTORICAL/NON-OPERATIVE.** This README preserves the
> dissolved SOL-ExecBench counter-free/B200 evaluation snapshot. Its nested
> `CLAUDE.md`, paths, nine-rule list, and surface descriptions are historical
> context only; they do not override root `CLAUDE.md`, active label schemas,
> `tools/local_labels/`, host runbooks, or accepted ADRs. Do not use this page to
> authorize NCU, clock, driver, privilege, or external-provider operations.

A counter-free GPU analytics framework for **NVIDIA's
[SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench)**
([paper](https://arxiv.org/abs/2603.19173)) — 235 GPU kernel
optimization problems on NVIDIA B200. The framework profiles complete
*.cu CUTLASS C++ / CuTe C++ / CUDA C++ template solutions hand-ported
from the official SOL-ExecBench Python references, using five
counter-free analytic surfaces (NVBit, CUPTI Activity, optional NVTX
under Nsight Systems, static introspection, compute-sanitizer)
that are **designed to migrate** from local dev to a non-root B200
provider — per-surface availability is recorded in the artifact
manifest rather than assumed.

**This project does not optimize kernels.** SOL Scores in the **0.3 –
0.5+** range are accepted for the template solutions — they are
profiling targets, not leaderboard contenders. The deliverable is the
framework + B200 profiling intel, not tuned kernels.

The retained historical contract snapshot is in [`CLAUDE.md`](CLAUDE.md); its nine
critical rules (no NCU, no clock-policy mutation, no privileged CUPTI, no
host-driver work, CUTLASS / CUDA C++ submission surface only, no evaluator-state
exploits, default-stream discipline, precision-is-contractual, IISWC paper
folder is operator-only), two-phase mission, five counter-free analytic
surfaces, architecture portability discipline, and submission format are not
current project policy. Root `CLAUDE.md`, active schemas, runbooks, and ADRs
control current operations.

## Two-phase mission

| Phase | Where | Goal |
|---|---|---|
| 1 | Local RTX 5060 (sm_120) | Hand-port the 235 problems into *.cu template solutions; verify the counter-free framework runs end-to-end. Local performance is **diagnostic, not a goal** — `speedup_factor < 1.0` is acceptable and the *.cu still lands. |
| 2 | B200 provider (sm_100a) | Migrate the framework as-is; re-run the five analytic surfaces over all 235 *.cu; upload solutions to the online judge for SOL Scores in the 0.3 – 0.5+ band; harvest the B200-specific profiling intel. |

## Layout

| Path | Purpose |
|---|---|
| `sol-execbench/` | Official NVIDIA evaluation framework (native `uv sync` install; CLI `sol-execbench`); produces correctness + speedup vs PyTorch reference |
| `SOLAR/` | NVlabs SOLAR — derives `T_SOL` hardware bounds via roofline + Orojenesis |
| `third_party/cutlass/` | Active unified-tree vendor path: official CUTLASS v4.6.1, scoped to `examples/`, `include/`, and `tools/`; see `third_party/cutlass.provenance.json` for the pinned commit/tree/digest. The upstream package's declared dependency remains separate; the build harness supplies `CUTLASS_DIR`. |
| `cutlass/` | Historical pre-dissolve path retained only in this archived description; it is not a live checkout dependency. |
| `baselines/` | The 235 *.cu template solutions, `baselines/{L1,L2,Quant,FIB}/<slug>/{cutlass,cuda_cpp}/` (see `baselines/README.md`) |
| `scripts/baselines_harness/` | Wraps the native `sol-execbench` CLI as a host subprocess; entry point for the five counter-free analytic surfaces |
| `scripts/baselines_scaffold/` | Emits `solution.json` + parent README boilerplate; lints forbidden surfaces |
| `nvbit_tracer/metric_extractor/` | Surface 1: non-root NVBit injector + metric extractor (per-warp histograms via JSONL + opt-in Tier-2 address-trace sidecar). |
| `references/NVIDIA_Manuals/solbench_problems/` | 235 problem specs (PyTorch reference `run()` is the behavioral contract) |
| `references/NVIDIA_Manuals/` | NVIDIA reference doc mirror + crawler scripts |
| `scripts/kb/` | Knowledge-base routing toolchain |
| `SOL-Execbench.pdf` | Benchmark paper (arXiv:2603.19173) |
| `AVO.pdf` | Agent-optimizer companion paper (reference context only) |
| `IISWC_conference_latex/` | **Operator-only black box; Claude does not read.** |

## Counter-free analytic surfaces

The framework drives the native `sol-execbench` CLI (installed via
`uv sync` of the in-tree `sol-execbench/` source tree) as a host
subprocess and layers five counter-free analytic surfaces over every
*.cu under `baselines/`.
Each surface is non-root, non-NCU, non-clock-mutating, and **designed
for** the local → B200 migration. Per-surface availability on the
chosen B200 provider is captured in the artifact manifest (`status:
ok | unavailable | blocked | failed | skipped`); the framework does
not assume every surface emits on every host.

1. **NVBit** per-warp metric histograms + opt-in Tier-2 address-trace sidecar
   (`nvbit_tracer/metric_extractor/metric_extractor.so` via `CUDA_INJECTION64_PATH=`)
2. **CUPTI Activity** (non-counter) wall-clock attribution per launch
   (`cuptiActivityFlushAll`)
3. **Nsight Systems + NVTX ranges** for application-level correlation
   (`nsys profile --trace=cuda,nvtx`)
4. **Static introspection** via `cuobjdump --dump-sass --dump-ptx`,
   `nvdisasm`, `ptxas -v` (build-time, no runtime dependency)
5. **`compute-sanitizer`** (memcheck / racecheck / synccheck / initcheck)

Artifacts land at
`scripts/baselines_harness/.harness_traces/<slug>__<lang>/{nvbit_metrics,cupti,nsys,static,sanitizer}/`.

## Quickstart

### 1. Install SOLAR (for `T_SOL` derivations)

```bash
cd SOLAR
bash install_uv.sh
source .venv/bin/activate
```

### 2. Install the sol-execbench evaluator natively

```bash
cd sol-execbench
./scripts/download_data.sh    # downloads the 235-problem dataset
uv sync --all-groups          # installs the sol-execbench package + CLI
source .venv/bin/activate     # or invoke commands via `uv run …`
```

This installs the `sol-execbench` CLI on `PATH` (no container build /
container entry needed). See `sol-execbench/README.md` for host
prerequisites (NVIDIA driver, CUDA toolkit, `uv`).

### 3. Sanity-check the official CUTLASS gemm example

```bash
sol-execbench examples/cutlass/gemm \
    --solution examples/cutlass/gemm/solution_cutlass.json
```

### 4. Hand-port a problem's *.cu template solution

```bash
problem=problem_004_gemm_n128_k2048
$EDITOR references/NVIDIA_Manuals/solbench_problems/${problem}.md

# Read the PyTorch reference and any official Python solution
# (sol-execbench/examples/{pytorch,triton,cute_dsl,...}) that matches
# the algorithmic pattern, then write the CUTLASS / CUDA C++ port:
mkdir -p baselines/L1/${problem}/cutlass
$EDITOR baselines/L1/${problem}/cutlass/main.cpp     # PyTorch binding
$EDITOR baselines/L1/${problem}/cutlass/kernel.cu    # kernel + CUTLASS templates
$EDITOR baselines/L1/${problem}/cutlass/solution.json
$EDITOR baselines/L1/${problem}/README.md

# Local correctness check on sm_120 (speedup not gated):
python scripts/baselines_harness/run.py \
    --baseline baselines/L1/${problem}/cutlass
```

### 5. Run the framework's five analytic surfaces

Wired in Phase 1; expects the *.cu to compile and pass correctness.

### 6. Submit to the online judge (Phase 2)

After the framework has been migrated to a B200 provider and the
analytic artifacts harvested, upload `solution.json` to
<https://research.nvidia.com/benchmarks/sol-execbench>. Acceptance
band: **SOL Score 0.3 – 0.5+**.

## Measurement reality check

The local RTX 5060 (sm_120) and the B200 provider (sm_100a) are
different Blackwell targets. A local pass proves compile / smoke
viability only — TCGen05, TMEM, TMA, cluster behavior, NVFP4 / FP8
layouts, register pressure, L2 partitioning, and HBM3e bandwidth are
all B200-specific surfaces that local sm_120 cannot validate. See
[`CLAUDE.md` → Architecture Portability Discipline](CLAUDE.md).

The local `sol-execbench` checkout's timing path diverges from the
paper's published methodology in subtle ways (CUPTI default in
`core/bench/timing.py`, 600 s CLI timeout default, dynamic L2 sizing).
Definitive `Tk` for the SOL Score comes from the online judge — never
from this local checkout.

## Per-problem development cycle

1. Read the problem spec in `references/NVIDIA_Manuals/solbench_problems/`
   (shape table + reference PyTorch `run()` + per-workload SOL +
   `Scoring Baseline` + `Reference Implementation` latencies).
2. Hand-port the PyTorch reference (and any matching official
   Python solution under `sol-execbench/examples/`) into a *.cu
   CUTLASS C++ / CUDA C++ template under
   `baselines/<category>/<slug>/{cutlass,cuda_cpp}/`.
3. Compose `solution.json` per `sol-execbench/docs/solution.md`.
   `spec.target_hardware: ["B200"]` for the cloud profiling target,
   `["LOCAL"]` (uppercase) for the 5060 correctness check.
4. Local correctness check on sm_120. A `speedup_factor < 1.0` is
   acceptable; correctness on every workload is required.
5. Run the five counter-free analytic surfaces (framework deliverable).
6. Migrate the framework to the B200 provider; re-run surfaces.
7. Submit `solution.json` to the online judge; record the returned
   SOL Score in the per-problem README.

## External references

- Paper: https://arxiv.org/abs/2603.19173
- Benchmark site: https://research.nvidia.com/benchmarks/sol-execbench
- Dataset: https://huggingface.co/datasets/nvidia/SOL-ExecBench
- SOL-ExecBench: https://github.com/NVIDIA/SOL-ExecBench
- SOLAR: https://github.com/NVlabs/SOLAR
- CUTLASS: https://github.com/NVIDIA/cutlass
- NVBit: https://github.com/NVlabs/NVBit
