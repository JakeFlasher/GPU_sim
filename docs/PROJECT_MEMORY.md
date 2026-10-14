# SM120 continuation memory

**Updated:** 2026-10-14

- Target: RTX 5090, `sm_120`; source `/root/GPU_simulator_rebase`; corpus
  `/root/sm120_rtx_5090_capture`; active plan is the proposal `PLAN.md`.
- Complete-corpus acceptance is real traces plus independent NCU comparison:
  339 roots and 11,558 launches per mode, full before fast, exactly 12 workers.
  NCU is never a replay input.
- Final effective-L1/protocol-split evidence is archived under
  `alpha_cleanup/final_12worker_closure_20261014`; flat alpha remains default
  because effective timing residuals and protocol-split residuals did not
  justify unconditional promotion.
- SM120 unified L1TEX metric projection uses sector-valid store/reduction terms;
  L2/DRAM remain unavailable. Effective L2 is a bounded 96 MiB/1-slice/16-way
  line-mod/LRU scenario with material residual and is not default.
- Landed source fixes separate TMA submissions from legacy async groups,
  preserve LDGSTS/TMA direction and barrier identity, reject fabricated counts,
  retain signed-operand precision loss, and keep diagnostic snapshots bounded
  and read-only. `completion_clock` remains unavailable.
- `sim-trace` streams plain or `.jsonl.zst` sidecars. Replay/comparison tools
  stream `.jsonl.zst`, enforce 12 workers, clean tracer preload variables,
  bound child output, and fail closed on malformed source/output.
- Conditional flags remain explicit because they name model assumptions, not
  hidden physical truth. Do not import external SOTA topology constants without
  target-scope evidence and corpus comparison.
- Temporary research is under `/dev/shm/sm120_alpha_tmp_20261013`; compact
  evidence is under `replay_results/`; historical review and superseded docs are
  under `adversarial_review/history/`. No fixture, hash, verification,
  admission, or Accel-Sim workflows.
- Proprietary scheduler/cache topology, TMA engine completion, universal clock
  conversion, and formal unseen-workload accuracy are documented non-prerequisites
  under the university PhD closure boundary.
- Post-remediation flat corpus run (commit `202132b6`) completed 339/339 roots
  and 11,558 launches in both full and fast modes with 12 workers; walls were
  84.541 s and 57.453 s. The async diagnostic snapshot cap was reduced to 1,024
  after an earlier large grouped-GEMM report exceeded the driver output bound.
