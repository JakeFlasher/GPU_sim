# SM86 MUFU/SFU frontier probe

This is an independent, unscored clock probe for the RTX 3090 Ti (SM86). It
measures compile-time-specialized one-chain and four-chain `MUFU.EX2` bodies at
several resident-warp counts. The chain count is a template parameter, so a
runtime predicate cannot leave an unintended second MUFU body in the trace.
The output is evidence for an architecture capability; it is not a fit against
any scored workload or NCU target.

Build and smoke-test:

```bash
make -C workloads/research_probes/sfu_frontier
workloads/research_probes/sfu_frontier/sfu_frontier 84 1 1024 dependent
workloads/research_probes/sfu_frontier/sfu_frontier 84 8 1024 quad
```

For a locked reference point, use the project protocol (`1350 MHz` SM and
`10501 MHz` memory) before running the probe, and record the GPU UUID, driver,
clock state, command, compiler, and SASS. No scored simulator target is read
by this probe.

The bounded driver accepts signed-decimal arguments only, caps the chain at
262,144 iterations, rejects allocations above 64 MiB or half of free device
memory, initializes output sentinels, and reports timestamp/checksum presence.
The checksum is a device-produced MUFU result (not a bit-exact host
reimplementation); treat it as an output-write invariant, not a numerical
reference oracle.
