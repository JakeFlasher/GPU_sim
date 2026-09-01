# Completed milestone: sm_86 STORE acknowledgement frontier assay

**Status:** complete 2026-09-01 as
`complete_multicta_shift_transfer_no_per_store_timing`  
**Event implementation:** `0041edccda6de152aadeefb74db2276a57e4e220`  
**Assay implementation:** `38ced4a52b5028127ce9c5450ce3244083729b9e`  
**Reducer:** `5754535860c5b76214b6413f4105cfdbed0c3c6c`  
**Evidence:** `results/captures/sm86_store_ack_frontier_20260901_v1`  
**Tracked summary:** `data/local_labels/sm86_store_ack_frontier_report.json`

## Outcome

A fresh single-binary live assay independently confirms the causal shape that
the simulator's new write-ack event represents. With 84 one-warp CTAs,
outstanding scalar-global STORE work appears after the last in-kernel timestamp
in the ordinary same-stream handoff when no fence is present. Adding
`MEMBAR.SC.GPU` moves that delay into the explicit fence and contracts the
post-fence handoff.

The result transfers from depth 4 and 16 identification cells to an untouched
depth-64 holdout. It does **not** identify a per-store acknowledgement latency:
the pinned RTX 3090 Ti exposes `%globaltimer` in 1,024 ns quanta, and every
single-CTA cell remains at or below that resolution. The retained conclusion
is therefore mechanism-level and multi-CTA only.

Candidate 201 plus the event plumbing predicts the B84/D16 no-fence residual
within 6.54%. On the untouched B84/D64 trace it differs by 823 ns, less than
one observed timer quantum, although the percentage error is 40.19%. Its
device-fence error is 29.34% at both depths. This supports retaining the
candidate as an opt-in incomplete lower-bound mechanism, not promoting it to
default timing or complete retirement.

## Assay mechanism

One runtime-selected producer kernel contains three arms:

- `control`: identical address/checksum work, target STGs predicated off;
- `store_none`: target STGs active, no explicit fence; and
- `store_device`: the same target STGs followed by `membar.gl`.

Each CTA is one warp. Lane zero records the target-specific NVIDIA
`%globaltimer` before the payload loop, after STORE issue, and after the
optional fence. A common sentinel kernel is already queued behind the producer
on the same explicit nonblocking stream; its first body action records
`%globaltimer`, then it consumes one payload word. A timer-only 250-us kernel
precedes the pair and every trial proves host enqueue finished before that gate
ended, preventing small control kernels from measuring host submission delay.

CUDA documents ordinary same-stream operations as sequential and documents
implicit fence/flush work at task boundaries. PTX defines `%globaltimer` as a
64-bit global nanosecond timer while warning that it is target-specific:

- [CUDA stream ordering](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)
- [CUDA memory synchronization domains](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html)
- [PTX `%globaltimer`](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)

The sentinel interval is still composite: producer epilogue, common timestamp
stores, implicit task-boundary visibility, and sentinel dispatch. Only matched
arm differences are interpreted.

## Live evidence

The host stayed pinned to device 0, RTX 3090 Ti UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6, driver
610.57.04, P0, persistence enabled, and 1,350 MHz graphics/SM clocks.

The canonical campaign contains:

```text
10 correctness cells
150 independent native processes
13,950 raw paired-arm trials
361,305 raw CTA timestamp records
18 exact NVBit captures / 5,279,292 traced instructions
72 deterministic simulator JSON outputs
0 canonical failure files
```

All ten correctness cells passed. Every B84 trial observed 84 distinct `%smid`
values, every queue gate covered host enqueue, and every STORE sentinel
read the expected final payload.

Median matched observations are:

| Cell | STORE issue | Loaded GPU fence | No-fence handoff | Handoff contraction | Pair delta |
|---|---:|---:|---:|---:|---:|
| B84/D4 identification | 0 ns | 2,048 ns | 1,024 ns | 1,024 ns | 1,024 ns |
| B84/D16 identification | 0 ns | 4,096 ns | 3,072 ns | 3,072 ns | 3,072–4,096 ns |
| B84/D64 holdout | 7,168 ns | 4,096 ns | 2,048 ns | 2,048 ns | 9,216 ns |

For B84/D16, all 15 independent processes report the 4,096 ns fence and
3,072 ns no-fence handoff; all 15 contract by at least 3,072 ns after the
fence. For B84/D64, all 15 report the 7,168 ns issue component, 4,096 ns fence,
2,048 ns no-fence handoff, and 2,048 ns contraction. B1 depth, lane, reuse, and
gap variants are retained as resolution controls rather than averaged into a
false sub-microsecond claim.

## Exact trace and simulator bridge

The trace binary is spill-free and its SASS orders:

```text
GLOBALTIMER -> predicated target STG -> GLOBALTIMER
            -> MEMBAR.SC.GPU; ERRBAR; CCTL.IVALL
            -> GLOBALTIMER -> reporting stores
```

The 18 captures cover B1/D4, B84/D16, and the opened B84/D64 holdout, all three
arms, and two independent repeats. Every capture has exact total/reported
instruction equality and empty unsupported, unmapped, and cross-allocation
sidecars. The retained tracer runtime manifest validates with 55 DSOs.

Reporting/timestamp stores are common and remain visible in the trace. Matched
subtraction recovers exact payload work:

| Cell | Control STORE inst/sectors/acks | STORE-arm inst/sectors/acks | Exact payload delta |
|---|---:|---:|---:|
| B1/D4 | 5 / 12 / 5 | 9 / 140 / 9 | 4 / 128 / 4 |
| B84/D16 | 420 / 1,008 / 420 | 1,764 / 44,016 / 1,764 | 1,344 / 43,008 / 1,344 |
| B84/D64 | 420 / 1,008 / 420 | 5,796 / 173,040 / 5,796 | 5,376 / 172,032 / 5,376 |

Every default and write-ack replay has a byte-identical twin. Schema 22 drains
all scheduled acknowledgements, the STORE arms have identical traffic work,
and `dirty_writeback=true` plus `coherence=true` remain asserted.

At the observed 1,350 MHz SM clock:

| Cell | Live no-fence handoff | Simulated tail | APE | Live GPU fence | Simulated issue shift | APE |
|---|---:|---:|---:|---:|---:|---:|
| B84/D16 | 3,072 ns | 2,871 ns | 6.54% | 4,096 ns | 2,894 ns | 29.34% |
| B84/D64 holdout | 2,048 ns | 2,871 ns | 40.19% | 4,096 ns | 2,894 ns | 29.34% |

The repeated simulator values are identical because the trace and candidate
are deterministic; the differing live residuals are retained, not fitted away.

## Reproduction

```bash
.venv/bin/python \
  workloads/research_probes/store_ack_frontier/campaign.py build
.venv/bin/python \
  workloads/research_probes/store_ack_frontier/campaign.py static
.venv/bin/python \
  workloads/research_probes/store_ack_frontier/campaign.py correctness \
  --trials 3
.venv/bin/python \
  workloads/research_probes/store_ack_frontier/campaign.py native \
  --stage identification --repeats 15 --trials 31
.venv/bin/python \
  workloads/research_probes/store_ack_frontier/campaign.py native \
  --stage calibration --repeats 15 --trials 31
.venv/bin/python \
  workloads/research_probes/store_ack_frontier/analyze.py \
  --stage identification
.venv/bin/python \
  workloads/research_probes/store_ack_frontier/campaign.py native \
  --stage holdout --repeats 15 --trials 31
.venv/bin/python \
  workloads/research_probes/store_ack_frontier/campaign.py nvbit --repeats 2
.venv/bin/python \
  workloads/research_probes/store_ack_frontier/campaign.py replay --repeats 2
.venv/bin/python \
  workloads/research_probes/store_ack_frontier/analyze.py \
  --stage complete --validate-only
```

The holdout NVBit cell was opened only after the identification report passed;
use `--cells b84_d64_l32_line_unique_g0` for that capture/replay step.

## Scientific boundary and next action

Do not derive a per-store constant from the 1,024 ns timer buckets, use the live
values to retune candidate 201, or clear a physical tail. A same-stream
sentinel proves device-visible downstream consumption, not dirty eviction to
DRAM, complete coherence propagation, host/system visibility, or RED RMW.

The write-ack event plumbing is retained. Candidate 201 remains opt-in and the
completion frontier remains `incomplete_unmodeled_retirement_tails`. The next
GPU-simulator mechanism should be an explicit dirty-eviction/writeback stream
sharing a finite below-L2 write service; it must be identified independently
of this fence/terminal assay and of NCU target values.
