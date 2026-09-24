# Accel-Sim derived SM86 overlap probe

Updated 2026-09-24. This is a bounded architecture-alignment experiment, not
an official Accel-Sim validation or SOTA result.

## Derived profile

The native Accel-Sim `SM86_RTX3070` card was copied outside the source tree and
only these device-scale fields were changed:

| Field | Native value | Derived value |
| --- | ---: | ---: |
| `gpgpu_n_clusters` | 46 | 84 |
| `gpgpu_n_mem` | 16 | 24 |
| `gpgpu_clock_domains` | 1132:1132:1132:3500.5 | 1350:1350:1350:10501 |

All other scheduler, cache, memory-timing, and pipeline fields remain the
native RTX 3070 configuration. The profile is therefore **derived**, not an
official or independently validated RTX 3090 Ti card.

## Replay

Input: the native Accel-Sim legacy trace from the same GEMM launch used in the
earlier overlap probe. The bounded comparator found 512 matching warps,
197,120 records, 5,242,880 effective lane instructions, zero PC/effective-mask
mismatches, and zero normalized memory-payload mismatches against the enhanced
protobuf capture. The enhanced trace carries an additional shared-destination
record for each of 16,384 LDGSTS instructions; that is an additive contract
difference, not a different represented global source stream.

Three derived-profile replays completed:

| Run | Wall time | Peak RSS | Sim cycles | Sim instructions | CTAs | IPC | Exit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.419 s | 737,436 KiB | 12,588 | 5,635,808 | 512 | 447.7127 | 0 |
| 2 | 6.337 s | 737,220 KiB | 12,588 | 5,635,808 | 512 | 447.7127 | 0 |
| 3 | 6.299 s | 738,212 KiB | 12,588 | 5,635,808 | 512 | 447.7127 | 0 |

Artifacts are retained at
`/home/jakeshea/GPU_sims_notes/accel_overlap_20260924/derived_rtx3090ti/`.

## Validity boundary

The derived profile reduces the gross SM-count/clock mismatch, but it does not
turn the run into hardware accuracy evidence: inherited RTX 3070 cache,
scheduler, memory, and latency assumptions remain. Its `gpu_tot_sim_cycle`
counter is also not the local NCU `gpc__cycles_elapsed.max` observable. The
cell is recorded as `derived_profile_non_scoring`; no error percentage or
speed ranking is inferred.
