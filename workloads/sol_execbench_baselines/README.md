# SOLBench kernel sources

These directories contain CUDA/CUTLASS source inputs for real SOLBench traces.
They are workload inputs, not a correctness or admission test suite.

Use the complete RTX 5090 capture corpus under
`/root/sm120_rtx_5090_capture` and the Rust full-replay runbook in
`docs/runbook_host.md`. Independent NCU observations are joined after replay;
no local synthetic acceptance harness is maintained in this tree.
