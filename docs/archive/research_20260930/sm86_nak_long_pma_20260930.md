# Long independent NAK PMA body (2026-09-30)

**Count correction:** the 416/1,560 counts below counted eight-byte transport
chunks including headers, not decoded samples. Reassembly gives 361/1,364
complete samples, respectively. Use `sm86_nak_pma_coverage_20260930.md` and
`coverage_decoded_v2.json`; the original capture/summary files are retained.

The retained short NAK PMA body produced only 52 records. The new
`tools/sm86_nak_long_pma_capture.py` restores the public tinygrad/NAK producer,
generates a data-dependent integer/bitwise body, and captures direct PMA.

The 64-round variant contains 1,330 static SM86 instruction words and produces
1,560 PMA records (1.17 records per static word), with 916 unique raw records.
This is a substantial sampling-coverage improvement over the short body. An
attempted direct NCU `PmSampling` wrapper around the standalone NAK runtime
reported no profiled kernels. An additional NVIDIA CUPTI
`pc_sampling_continuous` injection built from the installed public sample also
produced no decoded file for the standalone NAK driver path. Only the direct
PMA stream is therefore retained for the long body. A second attempt using
direct `LD_PRELOAD` produced the same no-record outcome.

The raw stream still has no physical pipe/sub-unit, issue-slot, or exact dynamic
issue-cycle field. The result strengthens aggregate PC/stall-state modeling and
does not close the physical-pipe blocker.

Artifacts: `results/alpha/sm86_nak_long_pma_goal_20260930_variant/`.
No tests, hashing, or verification workflow was run.
