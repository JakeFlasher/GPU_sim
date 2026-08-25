# sm_86 fixed-latency register-file-cache probe

Five functions execute sixteen independent bare `IMAD` instructions per
generation. Eight factor registers are loaded before timing from distinct
addresses but contain the same value. A checked cubin patcher can therefore
change physical factor identities and reuse bits without changing any result.

| Protocol | Variant A | Variant B |
|---|---|---|
| `u10` | alternating misses, reuse off | alternating allocations with no hits |
| `u11` | paired misses | first read allocates, second read hits |
| `u12` | hit without retention, then miss | hit retained, then hit again |
| `u20` | same register remains in source position 0 | register moves to position 1 and position 0 is replaced |
| `u22` | opposite-bank position-0 request preserves the entry | same-bank position-0 request evicts the entry |

All selected instructions are canonicalized to an exact physical layout. Pair
differences are restricted to selected source-register fields and reuse bits;
opcode, destination, predicate, non-reuse controls, resource declarations,
instruction counts, and runtime values remain equal.

Build and run a smoke test:

```bash
make -C workloads/research_probes/fixed_latency_reuse
workloads/research_probes/fixed_latency_reuse/fixed_latency_reuse_driver \
  workloads/research_probes/fixed_latency_reuse/fixed_latency_reuse_a.cubin \
  u11 a 4 8192 7
```

