# sm_86 adversarial atomic topology and cross-SM service milestone

## Outcome

Two adversarial successors plus a pairwise collision closure supersede the
first atomic timing candidate. They
promote exact ATOMG/RED sector transactions, exact L2 request topology, and
exact CLOCK-bounded replay diagnostics. They reject an operation-only sector
rate, a per-SM-only atomic queue, ordinary L2 cache hashes as atomic-unit maps,
and portable 128-byte or page-offset contention rules. Atomic timing remains
inactive because the distributed atomic-unit address hash is not identified.

This conclusion is consistent with three external findings:

- the [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
  distinguishes requests, serialized wavefronts, 32-byte sectors, and L2
  requests carrying up to four sectors;
- [Maucher et al., *Are Your GPU Atomics Secretly Contending?*](https://publikationen.bibliothek.kit.edu/1000188141/170539767)
  show that adjacent words can be fast while nominally separate cache lines can
  still cross-contend, and explicitly warn that address effects vary by GPU;
- NVIDIA documents that [NVCC automatically performs warp aggregation in many
  atomic patterns](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/),
  explaining why compiler-known uniform addresses are not causal contention
  controls.

## In-SM alias campaign

`atomic_alias_adversarial` uses volatile runtime index tables followed by a CTA
barrier, so every address treatment shares one static cubin per operation and
word width. Its 138 cells cross:

- returning ATOMG and non-returning RED;
- 32- and 64-bit words;
- alias degrees 1, 2, 4, 8, 16, and 32;
- grouped and scattered lane assignments;
- dense, 128-byte, and 256-byte intra-warp placement; and
- cross-warp spacing 0, 128, 256, 4096, and 4100 bytes.

The retained evidence comprises 1,932 correctness executions, 278 general NCU
profiles, 34 SourceCounters profiles, 12 return-path profiles, 138 NVBit
captures with 486,668 records, and 138 byte-identical replay pairs. Maximum W16
NCU CV is 1.047%; MAD is 0.606%.

### Exact topology

Dense 32-bit alias degrees 1, 2, and 4 all generate four L2 sectors per warp
instruction, but generate one, two, and four L2 requests respectively. RED W16
CLOCK timing separates them at 5,278, 6,337, and 8,418 cycles. Sector count
alone is therefore insufficient.

The compact replay representation is nevertheless sufficient. Active-lane
same-word aliases appear as repeated 32-byte sector ids. Exact L2 request count
is reconstructed by grouping those ids into 128-byte lines and summing each
line's maximum sector multiplicity. Replay now carries both quantities and
rejects incoherent request/sector pairs before execution.

### Resolved local RED law

For alias degree below 32, W16 RED active cycles equal a static intercept plus
one cycle per L2 sector plus one cycle per L2 request. A full-warp single-word
RED has an eight-request-per-cycle fast path. This law has 0.0140% MAPE on long
W16 NCU active cycles, 0.355% MAPE on W16 trace CLOCK timing, and 1.837% MAPE
across the W4/W8/W16 latency-floor transfer.

The earlier 9/8-cycle RED sector rate was an accidental endpoint identity: one
sector cycle plus one-eighth of a full-broadcast request cycle. Across the
expanded dense ladder, the old ATOMG/RED candidate has 14.903% MAPE and 45.635%
maximum error and is rejected.

### Returning path remains unresolved

ATOMG must return every old value. The 64-bit alias-16 treatment is slower than
alias-32, and return/writeback counters do not reduce to the RED law plus one
constant. No scalar ATOMG return correction is retained.

## Multi-SM campaign

`atomic_multism_service` uses one-warp CTAs at 1, 4, 16, 42, 84, and 168
blocks. Every warp aliases one word internally; blocks share that word or use
per-block words. The 106-cell matrix includes coarse 0/128/256/4096/4100-byte
placement, direct/permuted address order, and a 27-point B84 fine sweep.

The evidence comprises 1,484 correctness executions, 162 NCU profiles, 14
SourceCounters profiles, 106 NVBit captures, 2,668,105 records in 7,678 CTA
payloads, and 106 byte-identical replay pairs. Placement is exact: B42 and B84
use 42 and 84 distinct SMs; B168 puts exactly two CTAs on each of 84 SMs.
Maximum expanded-cell NCU CV is 1.482%; MAD is 0.891%.

### Shared and address-sensitive service

At B84, all cells have identical ATOMG/RED instruction, request, and sector
counts. Yet shared-word versus 4096-byte spacing differs by 7.36× for ATOMG and
17.15× for RED. A per-SM-only queue has 45.6% MAPE and 99.3% maximum error.

Well-distributed B84/B168 RED bounds about 23.52 effective parallel atomic
units. This is a throughput bound, not a discovered hash. A 72-candidate sweep
over line/sector/word domains, 6/12/24/48 buckets, and the six retained cache
hashes selects line/48-bucket/XOR only at 42.1% fine-sweep MAPE and 73.7%
block-count transfer MAPE. Ordinary cache slicing is rejected for atomic timing.

The fine RED sweep is also architecture-specific. On GA102, 4096 and 4100 bytes
are nearly equal, so the page-offset penalty reported for an AMD GPU does not
transfer. A 256-byte stride often improves on 128 bytes, but nearby 252 and 260
bytes are much worse; no scalar spacing threshold is lawful.

## Implementation changes

1. `sim-trace` projects global atomics onto alias-expanded 32-byte transaction
   sectors while exact lane addresses are still available.
2. Replay reconstructs and carries exact per-instruction L2 atomic request
   count; malformed topology is a typed error.
3. Readiness schema 5 reports issued CLOCKLO reads, warps with a complete
   interval, and the maximum first-to-last CLOCKLO issue interval.
4. No timing queue, address hash, per-spacing correction, or ATOMG return
   constant is installed.

## Pairwise collision closure

The final oracle splits 84 CTAs evenly between two addresses. It sweeps every
aligned word below 128 bytes, every 128-byte line through 32 KiB, and every
four-byte absolute base phase for deltas 128, 256, 260, and 4096 bytes. Across
544 cells and 3,808 correct executions, one rule has zero classification errors:

```text
floor(address_a / 256) == floor(address_b / 256)
```

Same-region pairs form a fully separated high-contention cluster (median
155,280 cycles); different-region pairs form the independent cluster (median
69,157). `sim-arch` now exposes the 256-byte sm_86 cross-warp contention region
as a timing-free architecture capability. The result identifies the lock/
contention domain, not the many-region scheduling network: B84 throughput still
varies within the different-region class. No timing queue is activated.

The many-region successor now identifies maximum per-LTS RED-sector occupancy
as the causal local scheduling discriminator, but every tested address hash
fails transfer collision recall. Atomic timing therefore remains inactive. See
[many-region atomic/LTS scheduling](sm86_atomic_many_region_lts_milestone.md).
