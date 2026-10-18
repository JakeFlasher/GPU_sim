# SM86 local frame/coalescing endpoint — H102 (2026-10-16)

## Acceptance source

All accuracy numbers below come from the complete full-first replay of the
356-root corpus: 286 `ok`, 62 control dispositions, 8 replay errors, 12,392
full launches, exactly 12 simulator workers, compressed `.jsonl.zst` artifacts,
with NCU used only after replay for comparison.

* Replay: `../sm86_replay_results/full_corpus_h102_local_hybrid_20261016_143358/`
* Report: `../sm86_replay_results/metric_report_h102_local_hybrid_20261016_v1/`

## Mechanism

SM86 local SASS prologues in the three real local-memory roots allocate a
256-byte private frame. Dynamic-index local records retain stack-relative
offsets that collapse lane-private slots in the ordinary sector plan. H102
retains two source-derived candidates:

* transfer-width lane expansion (`local_canonical_*`), and
* fixed 256-byte frame expansion (`local_frame256_*`).

For a dynamic-index shape only (`raw sectors > requests`), the selected
observer endpoint is:

```
raw sectors + (frame256 sectors - transfer-width sectors)
```

This preserves the observed coalesced payload and adds only the missing
lane-private frame domain. Scalar volatile and register-spill shapes remain on
the payload candidate; no timing/cache state is changed.

## Full-corpus result

| metric | H99 before | H102 | status |
|---|---:|---:|---|
| local load sectors | 75.85% | **1.80%** | accepted |
| local store sectors | 78.14% | **0.34%** | accepted |
| local load lookup hits | 76.50% | **1.80%** | accepted |
| local load lookup misses | exact count 104/107 | **exact count 107/107** | accepted |

Every other scored metric is unchanged from H99, including aggregate L2 TEX,
LTC128B L1, DRAM, cycles, and IPC.

## Final-tree confirmation (H105)

After rolling back the rejected H103 selector and rebuilding the release
binary, the final-tree H105 full-first replay was run again:
`../sm86_replay_results/full_corpus_h105_final_release_20261016_145305/`,
report `../sm86_replay_results/metric_report_h105_final_release_20261016_v1/`.
It retained the H102 local numbers and the H99/H93 non-local numbers with the
same 286/62/8 and 12-worker coverage.
