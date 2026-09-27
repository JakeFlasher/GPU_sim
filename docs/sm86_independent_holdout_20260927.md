# Independent SM86 microbench holdout — 2026-09-27

This is a bounded live RTX 3090 Ti/NVBit/NCU holdout, not a publication-grade
timing score. It was run after the prior bounded campaign on a fresh device
identity and fresh microbenchmark output directories. No large workload or
external simulator was run.

## Host identity

```text
GPU: NVIDIA GeForce RTX 3090 Ti
UUID: GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c
Compute capability: 8.6
Driver: 615.71.09
Observed clocks before campaign: 210 MHz SM, 405 MHz memory
CUDA: 13.4.92
Nsight Compute: 2026.1.0.0
Capture/replay repository revision at measurement time:
`e9d750f6e21373659186013a50f9f595d30d9a3b`
```

The clocks were **not locked**, so these measurements are not used for an
elapsed-time accuracy claim. The device/driver tuple is recorded to prevent
confusing this capture with the older development corpus.

The prerequisite audit confirms why a locked repeat was not attempted: the
current user has no passwordless administrative privilege (`sudo -n true` is
not available), `nvidia-smi -rgc` and `-rmc` refuse clock changes with a
permission error, and `nvidia-smi -lmci` reports runtime memory-clock
switching. This is a typed environment blocker, not a silently relaxed clock
policy.

The prerequisite probe is preserved at
`results/captures/sm86_locked_clock_prerequisite_20260927_v1/` (ignored local
capture): host-state SHA-256 `dac85e57475671c7374fa795bdf93b6f0a364772e45063037cc114ec35d35254`,
timestamp SHA-256 `6a2b426c1b807f1bd59c64d524f774455ae53764f51c86526a6ceadb36467a9c`,
and permission-probe SHA-256
`fab7d2019102e13f1c664a5e23189e716da473ead02600d031f79e295ede7ea6`.

Raw ignored campaign directories (preserved locally) are:

```text
results/captures/sm86_independent_holdout_20260927_v1
results/captures/sm86_independent_holdout_20260927_writeback_v1
```

Content identities for the compact comparison artifacts:

```text
domains comparison.csv:   ab5cf1ef0eae6bcd32830c57bf726e43a0a507b467a7f86d2a738b640d1130f1
domains inputs.json:       f0fc80d759599a98b2e8559f41125722748868c0770ce9440cce122b113a024c
writeback comparison.csv:  0d6843baf5a49c6ff92273aac87a20f5683e0562dfd9b01add645beb1a0a5239
writeback inputs.json:      6feb192fe9f0a6413c0694220d3090281c5e1be9dfe8837082f6e291765bf812
```

## Independent observations

The campaign used one-block/one-warp kernels with read, atomic, reduction,
control, and store patterns. Every row had identical simulator and NCU warp
instruction counts. Ordinary global-store request/sector counts also matched;
atomic/reduction sector populations matched their NCU counterparts.

| Family | Cases | Simulator/NCU warp instructions | Simulator/NCU store sectors | Simulator/NCU atomic or reduction sectors |
| --- | ---: | ---: | ---: | ---: |
| Read | 3 | exact (55/55, 59/59, 62/62) | exact (4/4 each) | n/a |
| Atomic | 3 | exact (63/63, 67/67, 70/70) | exact (4/4 each) | exact (32/32, 256/256, 256/256) |
| Reduction | 3 | exact (53/53, 57/57, 60/60) | exact (4/4 each) | exact (32/32, 256/256, 256/256) |
| Control | 2 | exact (47/47) | exact (4/4) | n/a |
| Store/writeback probe | 2 | exact (209/209 each) | exact (36/36 each) | n/a |

## What did **not** match or is not scoreable

- NCU `dram__sectors_read.sum` is 92–132 on these runs while the simulator's
  modeled read stream is 10–50 sectors. This is an explicitly different metric
  domain/endpoint, not an accuracy percentage.
- The simulator reports `dram_write_status=unavailable` for these probes and
  no dirty L2 eviction occurred. They do not validate resident dirty flush or
  hardware writeback completion.
- Clocks were dynamic, and the sample is one-block/one-warp rather than a
  five-family, repeated, locked-clock holdout. The result is T2 **counter
  contract evidence only**, not T2 elapsed-time accuracy.

The NCU CSVs were collected independently before replay; they were not used to
select model parameters, branches, or implementation paths. The replay used
the content-bound binary/card/trace/control identities listed in each
`replay.inputs.json`.

## Next bounded hardware step

Repeat the five-family matrix with locked clocks, five repeats, explicit cache
state controls, and a separate flush/eviction kernel whose trace contract
includes all dirty sectors. Only then can timing/cache-domain errors be scored;
the current result must remain a non-scoring holdout.
