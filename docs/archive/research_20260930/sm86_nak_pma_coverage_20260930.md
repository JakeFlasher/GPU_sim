# Direct NAK PMA coverage model (2026-09-30)

`tools/sm86_nak_pma_coverage_model.py` v2 summarizes decoded PMA sample density
for three independently generated NAK bodies. It reassembles payloads per TPC
from 32-byte transport packets before counting eight-byte samples. The v1
`len(blob)//8` count included headers and padding and is superseded.

| body | static instruction words | PMA records | records/instruction |
|---|---:|---:|---:|
| short retained body | 10 | 52 | 5.20 |
| simple long body | 20 | 361 | 18.05 |
| data-dependent 64-round body | 1,330 | 1,364 | 1.03 |

These are counts under tinygrad's reverse-engineered decoder, not independent
silicon truth or dynamic instruction coverage. The framing reports 8/55/196
drop-marked packets, respectively, and per-stream trailing partial bytes.
Those flags are not a measured sample-loss count. No physical route or dynamic
issue ordinal has been identified; constant unassigned bits do not prove that
the format cannot carry additional information under other stimuli.

Report: `results/alpha/sm86_nak_pma_coverage_goal_20260930/coverage_decoded_v2.json`.
Historical captures and v1 output remain unchanged.
No tests, hashing, or verification workflow was run.
