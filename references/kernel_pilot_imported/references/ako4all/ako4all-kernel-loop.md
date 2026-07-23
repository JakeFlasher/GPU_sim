# AKO4ALL Kernel Loop

This reference adapts the AKO4ALL framework from the SGLang diffusion AKO4ALL skill for general AI-infra GPU kernel work.

## Minimum AKO Layout

Inside `AKO4ALL/<task>/`, prefer:

- `input/reference.*`
- `input/<kernel>.*`
- `solution/<kernel>.*`
- `bench/bench_<kernel>.*`
- `context/<kernel>_notes.md`
- `ITERATIONS.md`

Use the target stack's natural extension. **For solbench submissions: `.cu`/`.cc`/`.h` only — CUDA C++ / CUTLASS C++ / CuTe C++ (Rule 5). Triton / CuTe DSL Python `.py` files appear in this section only as algorithm references; rewrite the algorithm in CUDA C++ / CUTLASS C++ for the *.cu hand-port.**

Bootstrap solbench *.cu hand-ports from `scripts/baselines_scaffold/` (which emits `solution.json` + `baselines/<category>/<slug>/README.md` boilerplate and lints against the forbidden-surface catalog). The kernel-pilot `templates/` directory referenced by the original AKO4ALL loop (with `ITERATIONS.md`, `kernel_notes.md`, `bench_kernel.{py,cu}` templates) is NOT part of the solbench import — the solbench scaffolder owns the equivalent role and emits the canonical layout used by the 235 problem dossiers under `baselines/{L1,L2,Quant,FIB}/<slug>/{cutlass,cuda_cpp}/`.

## Baseline Checklist

- Reproduce the current target-repo implementation in AKO first.
- Confirm the AKO baseline matches the trusted reference.
- Run the microbench before making edits.
- Capture one representative [counter-free analytic baseline] for the hottest meaningful shape.
- Use `nsys` when the suspicion is launch overhead, stream gaps, or end-to-end scheduling.
- Record the plain-language bottleneck: registers, occupancy, instruction count, DRAM, L2, shared-memory bank conflicts, launch overhead, synchronization, or layout conversion.

## Iteration Discipline

- One optimization idea per iteration.
- Re-run correctness before timing.
- Re-run the benchmark after every code change.
- Log the hypothesis, command, result, and next decision in `ITERATIONS.md`.
- Keep the current best candidate easy to identify.

Stop a direction early when:

- 3 consecutive iterations do not beat the best runtime
- correctness becomes fragile or shape-specific
- the win depends on unrepresentative shapes
- the microbench improves but the real operator or model path does not

## Validation Gate

Before calling a kernel done, validate all of:

- syntax, import, build, or compile check
- focused correctness test
- kernel or operator benchmark
- old implementation vs new implementation when this is a rewrite
- model-level or service-level benchmark when the kernel is on a production inference path

## Artifact Checklist

Keep these artifacts with the final notes:

- microbench table
- operator or model benchmark table
- counter-free analytic before/after summary for the representative shape
- `nsys` summary if launch gaps or overlap were involved
- one short explanation of why the kernel got faster or why a direction failed
- source reference list for any copied or adapted technique
