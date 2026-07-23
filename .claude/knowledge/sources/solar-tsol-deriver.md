---
Use when: looking up the structure of the NVlabs SOLAR T_SOL deriver — the 5-stage-pipeline PyTorch-to-einsum-to-roofline-analysis toolchain whose `perf_<arch>.yaml` output produces the analytical Speed-of-Light bound (T_SOL) that anchors the denominator of the SOL Score formula.
Primary raw source: SOLAR/
Supporting sources: SOLAR/README.md; SOLAR/docs/USAGE.md; SOLAR/docs/SOL_GUIDE.md; SOLAR/docs/EINSUM_GUIDE.md; SOLAR/docs/EINSUM_GRAPH_CHECKER_GUIDE.md; SOLAR/docs/TESTING_GUIDE.md
Repo touchpoints: SOLAR/install.sh; SOLAR/install_uv.sh; SOLAR/configs/arch/B200.yaml; SOLAR/configs/arch/H100_PCIe.yaml; SOLAR/configs/arch/; SOLAR/solar/; SOLAR/examples/; SOLAR/tests/; SOLAR/patches/torchview-parameter-tensors.patch
Patents anchored here: None
Out of scope: This stub is a navigation index for the upstream codebase, not a re-authored SOL methodology manual. Do not edit `SOLAR/` — it is an upstream NVlabs clone tracked separately. T_SOL is produced by SOLAR; the SOL Score formula and the leaderboard's Tk-vs-T_SOL bookkeeping are owned by `sources/sol-execbench-paper.md` and the public scoring service, not by SOLAR itself.
Version family: solar-nvlabs-2026
Default skill: implementing-sol-baselines
Routing tags: tsol-deriver; orojenesis-elements; einsum-graph; einsum-graph-renamed; 5-stage-pipeline; arch-config-B200; perf-prediction; pytorch-graph; roofline-analysis; timeloop-export
---

# SOLAR T_SOL Deriver (Citation Stub)

## Use when

Looking up where a SOLAR pipeline stage lives, which input YAML feeds which analysis step, or which architecture config to point Stage 4 (perf-prediction) at when deriving a per-workload T_SOL value for B200 vs H100.

## Pointer

The upstream codebase lives at `SOLAR/` (cloned at the repo root). SOLAR = **S**peed **O**f **L**ight **A**nalysis for **R**untime. This card mirrors the upstream `SOLAR/README.md` plus `SOLAR/docs/USAGE.md` table-of-contents; consult the upstream files for authoritative details.

## Installation

- Quickstart — `bash install.sh` from `SOLAR/` (installs SOLAR plus the patched `torchview` from `SOLAR/patches/torchview-parameter-tensors.patch`). Alternative: `bash install_uv.sh && source .venv/bin/activate` for a uv-managed virtual environment.
- Optional — install Graphviz (the `dot` binary) for PDF graph rendering.

## The 5-Stage Pipeline

Per `SOLAR/docs/USAGE.md` § The 5-Stage Pipeline:

1. PyTorch Graph Extraction — `solar-process-model` → `pytorch_graph.yaml`.
2. Einsum Conversion plus Rank Renaming — `solar-toeinsum-model` → `einsum_graph.yaml`, `einsum_graph_renamed.yaml`, optional `einsum_graph.pdf`.
3. Hardware-Independent Analysis — `solar-analyze-model` → `analysis.yaml`.
4. Performance Prediction — `solar-predict-perf-model --arch-config B200|H100_PCIe` → `perf_<arch>.yaml`.
5. Timeloop Export (optional) — `solar-totimeloop` → `timeloop_graph.yaml`.

## SOL Methodology

Per `SOLAR/docs/SOL_GUIDE.md` — SOL = Speed Of Light Analysis for Runtime. The guide documents three roofline-based performance models the perf-prediction stage uses (Unfused / Fused / Fused+Prefetched variants), all currently applying a single whole-graph `max(compute_cycles, memory_cycles)` roofline; §6 (Known Implementation Gaps) notes that in the current implementation `fused` and `fused_prefetched` produce identical memory totals and therefore identical runtimes. The "orojenesis" terminology that appears in SOLAR's code (`orojenesis_elements`, `orojenesis_bytes`) is the legacy name for unfused-mode element / byte counts; `solar/analysis/graph_analyzer.py` sets `orojenesis_elements` to `None` when those legacy runs are not enabled, and `solar/perf/perf_model.py` falls back to the modern `unfused_elements` field.

## Einsum Conventions

Per `SOLAR/docs/EINSUM_GUIDE.md` and `SOLAR/docs/EINSUM_GRAPH_CHECKER_GUIDE.md` — einsum equation format, rank-variable naming, and the BFS-based dimension-label propagation produced by Stage 2's rank-renaming step.

## Architecture Configs

- `SOLAR/configs/arch/B200.yaml` — Blackwell datacenter B200 spec used by Stage 4 (HBM3e bandwidth, 5th-generation tensor cores, 1.5 GHz clock per the paper).
- `SOLAR/configs/arch/H100_PCIe.yaml` — Hopper H100 PCIe reference spec; SOLAR ships H100 as a non-Blackwell sanity-check arch.

## Repo Layout

Per `SOLAR/docs/USAGE.md` § Repo layout (high level):

- `SOLAR/solar/` — core library (`graph/`, `einsum/`, `analysis/`, `perf/`, `cli/`, `common/`, `benchmark/`).
- `SOLAR/examples/` — `Attention/`, `BERT/`, `Conv2d/`, `Matmul/` end-to-end runnable example models with per-example `run_solar.sh` drivers.
- `SOLAR/tests/` — pytest suite (see `SOLAR/docs/TESTING_GUIDE.md` for the `bash scripts/run_tests.sh quick` smoke runs).
- `SOLAR/configs/arch/` — hardware specs the perf-prediction stage reads.

## Critical caveat — T_SOL anchors the SOL Score denominator

The `perf_<arch>.yaml` output from Stage 4 contains the analytical per-workload latency (T_SOL) that anchors the denominator of the SOL Score formula `S(Tk) = (Tb − T_SOL) / ((Tk − T_SOL) + (Tb − T_SOL))` defined in project `CLAUDE.md` § Project Overview; SOLAR is the canonical T_SOL deriver, but the leaderboard's Tk-vs-T_SOL bookkeeping is owned by `sources/sol-execbench-paper.md` and the public scoring service, not by SOLAR itself.
