# Completed milestone: sm_86 L1TEX arrival and output-replay service boundary

**Status:** complete 2026-08-18; topology extended, timing service rejected  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[memory-retirement service boundary](sm86_memory_retirement_service_milestone.md)  
**Evidence:**
`results/captures/sm86_l1tex_arrival_replay_service_20260818_v1/analysis/README.md`

## Goal

Determine whether a transferable L1TEX service can be identified from ideal
wavefront work alone, whether it is per-SMSP or SM-shared, and whether observed
output replay can be priced as additional completion work. The discriminator
holds total ideal work fixed while independently varying warp concurrency, CTA
partition, arrival skew, active lanes, and 128-byte line topology.

## Probe construction

`workloads/research_probes/l1tex_arrival_replay_service` uses one to eight
warps per CTA. Every warp executes 256 common dependent IMAD instructions,
rendezvous, optionally executes a warp-indexed dependent gap, and then performs
a control, discarded-load, store, or RED loop.

The primary cells all carry 512 ideal wavefronts per nominal SM:

- W1×depth64, W2×depth32, W4×depth16, and W8×depth8;
- 1 CTA×8 warps, 2 CTAs×4 warps, and 4 CTAs×2 warps per nominal SM;
- line-separated 4/8/16/32-lane variants with inversely scaled depth; and
- a full-lane word-coalesced depth64 variant.

Additional W1/W4 work scaling and W4/W8 arrival gaps of 16/64/256 dependent
steps identify phase and overlap behavior.

## Complete evidence

- 75 cells and 225 correct executions;
- 341 general NCU profiles after targeted five- and seven-repeat expansion;
- 12 SourceCounters profiles and 20 focused L2-request profiles;
- 75 NVBit captures with 159,553,212 records;
- exact total/reported trace cardinality and empty unsupported, unmapped, and
  cross-allocation sidecars in every capture;
- four spill-free kernels, 320 decoded instructions, and zero unknown
  encodings; and
- 75 ordinary plus 75 carded-DRAM deterministic replay pairs.

Maximum retained GPC/SM-active CV is 10.517%/8.634%. High variability is
localized to line-stream service phases and remains explicit.

## Adversarial construction correction

The first reducer pass caught a host-only invariant bug: it initially computed
word-coalesced ideal work using the line-separated lane formula. Schema 6 had
already projected the trace correctly—four word sectors in one line form one
ideal wavefront—but the JSON checksum claimed eight. The device SASS and all
NCU/NVBit measurements were unaffected. The host formula was corrected,
binaries rebuilt, all 225 correctness executions regenerated, static artifacts
refreshed, and the complete analysis rerun. The word cell now enters the
fixed-512 comparison correctly.

NCU application replay collects metric groups on different executions, so
cross-metric associations are replicated cell-level evidence rather than
co-temporal mediation. The template-specialized target and control kernels
also differ in SASS and register count; matched subtraction is therefore a
whole-kernel marginal cost, not the isolated latency of one memory opcode.

## Fixed-work concurrency rejects a work-only rate

At exactly 512 ideal wavefronts per nominal SM:

| tail | W1 | W2 | W4 | W8 | direction |
|---|---:|---:|---:|---:|---|
| load GPC delta | 39,212 | 23,495 | 16,816 | 14,049 | 2.79× faster with concurrency |
| store GPC delta | 17,074 | 20,013 | 21,429 | 21,873 | 1.28× slower with concurrency |
| RED GPC delta | 29,412 | 33,726 | 34,256 | 33,548 | nonmonotonic plateau |

Load latency is hidden by more resident warps. Store issue pressure moves in
the opposite direction. RED remains coupled to its separate atomic network.
One ideal-wavefront rate cannot explain all three.

Store output replay also moves independently: extra wavefronts fall from
1,344 at W1 to 247 at W8 even while store time rises. It is not a positive
per-wavefront delay.

## CTA partition identifies an SM-level timing relation but not replay count

Keeping eight warps and 512 ideal wavefronts per nominal SM, 1×8, 2×4, and 4×2
CTA shapes remain close:

| tail | maximum range across shapes |
|---|---:|
| load | 269 cycles |
| store | 877 cycles |
| RED | 510 cycles |

Simulator maximum issue spans likewise remain nearly unchanged. This supports
an SM-level work relation over CTA-local service. But store extra replay changes
247→894→738 while time remains close, again rejecting replay count as a scalar
latency input.

## Unique 128-byte lines are an essential work dimension

Readiness schema 6 now additionally reports, per operation class:

- exact unique 128-byte lines; and
- exact ATOMG/RED L2 requests, distinct from both lines and ideal wavefronts.

Twenty focused NCU profiles validate projected load/store unique lines against
`lts__t_requests_srcunit_tex_op_read/write.sum` and RED request topology
against `lts__t_requests_srcunit_tex_op_red.sum` with zero mismatches.

The fixed-ideal topology is decisive. Store line arms with 4/8/16/32 active
lanes all preserve 512 ideal wavefronts and 2,048 unique lines per SM and stay
within 923 cycles (20,950–21,873). The word arm preserves 512 ideal wavefronts
but has only 512 unique lines and falls to 6,792 cycles. Ideal wavefronts alone
discard required address topology.

Load retains an additional instruction/latency-hiding dimension: line arms with
the same 512 ideal wavefronts and 2,048 lines range from 14,049 to 37,791 cycles
as per-warp depth changes.

## Arrival skew rejects summary-span and replay-delay models

For W8 store, increasing the warp gap from 0 to 256 reduces the live tail from
21,873 to 2,047 cycles while simulator memory-issue span grows from 274 to
41,493 cycles. During the same change, extra output replay rises from 247 to
1,738 wavefronts. W8 load similarly falls from 14,049 to 5,318 without output
replay.

The actual distribution of arrivals within the interval matters. A first/last
span cannot reconstruct how much work is served before the last warp arrives.

## Candidate rejection

Affine surfaces over `(ideal wavefronts, unique lines)` were trained only on W4
work scaling plus the word-coalesced cell. Store fits that narrow slice at
166.9-cycle MAE, but held-out concurrency/CTA/skew/topology MAE is 4,617.6 and
maximum error is 19,395.6 cycles. Load training MAE is 550.5; held-out MAE is
6,882.5 and maximum error is 23,632.1 cycles.

No work-only, per-SMSP-only, SM-shared fixed-rate, affine-work, or
output-replay-delay model is promoted.

## Promotion decision

Promoted without timing changes:

- scalar-global unique 128-byte line counts;
- exact atomic/reduction L2 requests in the schema 6 issue diagnostic;
- checked cross-SM aggregation; and
- direct JSON serialization alongside instructions, sectors, ideal
  wavefronts, and issue spans.

Still inactive:

- L1TEX issue/service delay;
- output replay prediction;
- store/RED completion events;
- RED many-region scheduling; and
- numerical GPC-cycle emission.

That prerequisite is now complete in the
[bounded streaming candidate milestone](sm86_streaming_memory_service_candidates_milestone.md).
Schema 7 evaluates at each actual issue with fixed memory and zero production
scheduling effect. An SM-shared line-service hypothesis transfers
retrospectively, but rates remain inactive pending a prospective finite-queue
and issue-backpressure campaign.

No privileged command was required.
