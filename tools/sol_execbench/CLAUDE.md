# SOL-ExecBench source notes

This checkout is retained as the upstream SOLBench evaluator and workload
source.  Use `uv sync --all-groups` to provision it and invoke
`uv run sol-execbench PROBLEM_DIR --solution solution.json` for a real workload
execution.

The simulator alpha does not use this package's small fixture or test suites as
acceptance evidence.  Capture real SOLBench workloads into
`/root/sm120_rtx_5090_capture`, then run complete-corpus Rust replay and join
independent NCU observations afterward.

The package layers are `cli/`, `driver/`, and `core/`; source changes should
preserve the upstream evaluator's JSONL trace contract.
