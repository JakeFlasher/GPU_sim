# SOLBench kernel sources

These directories contain CUDA/CUTLASS source inputs for real SOLBench traces.
They are workload inputs, not a correctness or admission test suite.

Use the complete capture corpus supplied to a run and the Rust full-replay
runbook in `docs/REPLAY.md`. Independent NCU observations are joined after replay;
no local synthetic acceptance harness is maintained in this tree.
