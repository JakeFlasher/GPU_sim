# GPU and SOL knowledge index

This tree retains the project knowledge corpus for direct simulator work and
future opt-in SOL/campaign work. Cards are reference material; they do not grant
or withhold permission to run tools or continue research.

## GPU reference cards

- `ptx/` and `inline-ptx/`: PTX instructions and inline assembly.
- `sass/`: SASS decoding and NVIDIA binary tools.
- `cuda/` and `best-practices/`: memory, scheduling, synchronization,
  occupancy, and kernel optimization.
- `nvcc/`: CUDA compilation and fatbin behavior.
- `blackwell/`: `sm_100` and `sm_120` compatibility and tuning.
- `measurement/`: NCU, CUPTI, NVBit, device timers, and interpretation of
  historical measurement regimes.
- `simulator/`: Rust discrete-event implementation patterns and simulator
  prior art.
- `patents/`: NVIDIA architecture history and provenance.

## Retained SOL/campaign cards

- `problems/`: per-problem SOL-ExecBench dossiers.
- `patterns/`: CUDA/CUTLASS/CuTe implementation patterns.
- `sources/`: source and paper routes.
- `cutlass_constructs/`: reusable CUTLASS/CuTe concepts.
- `bitlesson.md`: historical engineering lessons. Its technical defect reports
  remain useful; its old round, audit, gate, ledger, and RLCR procedures are
  retired under this directory's `AGENTS.md`.

Use the task-shaped skills under `.claude/skills/` to select a small relevant
set. `run-sol-campaign` is the opt-in entry point for the preserved SOL and
campaign stack. No generated routing database or prompt hook is required.
