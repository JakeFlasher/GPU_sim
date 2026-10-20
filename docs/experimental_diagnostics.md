# Diagnostic and functional-model boundaries

**Updated:** 2026-10-13

The native path consumes real NVBit traces, static control sidecars, and typed
async observations. It reports issue/readiness, dependency, memory, async,
launch, and replay-process counters without inventing unobserved hardware events.
Full replay is primary; the core-only fixed-latency mode is a complementary
diagnostic.

Optional Rust seams expose carded L1/L2/DRAM service, resident-L1 visibility,
TMA/LDGSTS payload and barrier semantics, and bounded functional SIMT programs.
They are mechanism probes, not a claim of vendor-equivalent timing. Malformed or
unsupported inputs fail closed in production code.

Keep these timing domains separate:

1. model cycles and named completion timestamps under the declared architecture
   card;
2. replay-process wall time (host parsing, scheduling, and serialization);
3. uninstrumented CUDA-event time for the same launch scope;
4. independent NCU duration and counters.

When a domain is unavailable, report it as unavailable rather than substituting
another clock. The complete real capture corpus supplied to a run is the
performance population. NCU is joined after replay and never becomes a
simulator parameter.
