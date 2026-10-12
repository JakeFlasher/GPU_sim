# SOLBench workload sources

This directory holds CUDA source and launch-shape definitions used to produce
real SOLBench traces.  It is a source tree, not an acceptance-test suite.

## Active workflow

1. Build the selected real workload with the host CUDA toolchain.
2. Capture NVBit protobuf traces and typed sidecars into the RTX 5090 corpus.
3. Replay the complete corpus with the Rust simulator (full mode first).
4. Join independent NCU observations after replay and report units, scope,
   joins, and residual gaps.

Do not add local-label admission ledgers, fixture bundles, golden outputs, or
small synthetic gates here.  Runtime parsers may reject malformed input, but
those errors are implementation behavior rather than evidence gates.

The top-level roster and shape files describe workload inputs only.  The active
simulator runbook is [`../../docs/runbook_host.md`](../../docs/runbook_host.md).
