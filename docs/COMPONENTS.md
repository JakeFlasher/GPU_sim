# Component inventory

| Component | Keep and develop | Cleanup direction |
|---|---|---|
| `sim/engine` | Event scheduling and actor batching | Benchmark event throughput |
| `sim/trace` | Protobuf decode and trace semantics | Prefer direct paths and bounded reads |
| `sim/sm` | Issue, dependency, replay, and control-bit logic | Complete cycle semantics |
| `sim/mem` | L1/L2/DRAM behavior | Calibrate from microbenchmarks |
| `sim/direct_run` | Default end-to-end simulator application | Keep the CLI simple |
| `sim/run` | Optional historical campaign runner | Retain behind explicit package name |
| `tools/nvbit_tracer` | Tracer sources, schemas, reader, focused tests | Remove build/capture permission wrappers |
| `tools/local_labels` | NCU command driver, parser, metric catalog | Remove lifecycle and approval machinery |
| `workloads` | Kernels and ordinary experiment scripts | Remove orchestration ledgers |
| `eval` | Numerical prediction-versus-observation scoring | Add cohort import as needed |
| `arch_profiles` | Explicit architecture/SKU parameters | Keep unknown values visible |
| `tools/sol_execbench` | Optional SOL-ExecBench source, environment, and dataset | Retain in full; invoke explicitly |
| `tools/sol_baselines_harness` and campaign tools | Optional historical campaign capability | Keep scoped; never make it a root gate |

Generated build products, session worktrees, prompt packages, review receipts,
and duplicated historical plans are not project components. The retained SOL
environment, downloaded SOL dataset, and campaign corpus are explicit
exceptions because they are preserved research capabilities.
