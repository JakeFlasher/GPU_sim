# Long-term session notes

## Workspace and safety

* Work only in `unified_gpu_simulator`; `/home/jakeshea/5090_branch/` is
  strictly read-only.
* Supported cards are SM86 / RTX 3090 Ti and SM120 / RTX 5090. Card identity
  keeps their modeling assumptions separate.
* NCU profiles are independent targets joined after replay; never use them as
  inputs, tuning values, or workload-specific exceptions.

## Replay contract

* Accuracy claims use full integrated `l1-l2` replay with all supported features
  enabled, twelve workers, and compressed `.jsonl.zst` output.
* Do not run fast replay for final accuracy. Do not run the SM120
  DRAM-disabled full lane for acceptance.
* Required status profiles are SM120 339/339 `ok` and SM86 286 `ok`, 62
  `control_error`, 8 `replay_error` of 356 attempted.

## Current verified result

* SM120: global L1 hit 1.5567%, global L1 miss 0.5211%, unified L1 1.6386%,
  operation-read L2 6.6859%, generic TEX-L2 17.7364% Mean APE with 51 point
  zero-target FPs and a 99.81%/99.46% conditional packet envelope.
* SM86: operation-read L2 4.3416%, generic L2 8.5239%, and unified L1 9.87%
  on exact identity joins; ordered fallback is reported separately.
* HOTL-vs-Fenwick WAPE is 0.8767%/0.4462% (SM120 line/sector) and
  1.0223%/0.5324% (SM86 line/sector), with zero structural errors.
* IPC residuals are SM120 active 9.45% / elapsed 13.60% and SM86 active
  18.56% / elapsed 11.88%; see `docs/RESIDUAL_FRONTIER.md` for information
  barriers rather than speculative fixes.

## Scientific invariants

* Active event state (`SmCore`, `L1Cache`, `L2Model`, `DramServer`) is separate
  from report-only counter attribution and trace-locality observers.
* Footprint, lifetime, reuse distance, and miss-ratio curve terms follow
  Ding--Denning/Xiang terminology. A WAPE is weighted absolute error over
  finite MRC points; MAPE excludes zero targets.
* Conditional intervals are uncertainty-aware model outputs, not hardware
  truth. No count balancing, clamping, or kernel-name rule is acceptable.
* Normal policy rejects any regression in any metric. A trade-off requires a
  much larger target improvement, explicit evidence, and review.

## Release references

`docs/RESULTS.md` is the authoritative evaluation table;
`docs/ARCHITECTURE.md` and `docs/COMPONENTS.md` are the codemap;
`docs/TERMINOLOGY.md` is the vocabulary contract; `docs/HISTORY.md` holds
superseded campaign diaries. The closure plan is
`CACHE_LOCALITY_EXECUTION_PLAN.md`.
