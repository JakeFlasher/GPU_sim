# sm_86 SMSP scheduler-contention probe

This probe holds a 16-operation hot body fixed while placing one through four
warps in each `warpid_sm % 4` replay partition. It separates same-family
contention, cross-family order, and repeatable warp-cohort phase without global,
shared, or local memory, CTA barriers, asynchronous copy, or atomics inside the
measured interval.

| Protocol | Hot body |
|---|---|
| `q00` | phase/control only |
| `q10` / `q11` | dependent / four-stream carry-linked add |
| `q12` / `q13` | dependent / four-stream `IMAD` |
| `q14` / `q15` | dependent / four-stream `LOP3` |
| `q20` / `q21` | carry then IMAD / reversed by warp cohort |
| `q22` / `q23` | carry then LOP3 / reversed by warp cohort |
| `q24` / `q25` | IMAD then LOP3 / reversed by warp cohort |

Nominal phases 0, 4, 8, and 16 insert that many warp-vote issue slots for each
cohort depth on every loop generation. Warp-uniform forward branches skip those
slots for earlier cohorts. The vote result is kept live through the output
checksum but never enters the hot body. The labels are not assumed to be physical cycles:
the campaign records their complete SASS control words and subtracts matching
`q00` controls. NVIDIA documents `nanosleep` as approximate, so it is
deliberately not used as a phase oracle.

The long binary executes 8,192 generations; the bounded NVBit binary executes
64. Lane zero of every warp reports `clock64()` elapsed cycles and a host-checked
result.

```bash
make -C workloads/research_probes/smsp_scheduler_contention
workloads/research_probes/smsp_scheduler_contention/smsp_scheduler_contention \
  q20 8 8 7
```

The campaign matrix is frozen only after the build, correctness smoke, SASS
family/count audit, and pilot NCU profiles succeed.

The frozen 99-cell transfer matrix is resumable by stage:

```bash
.venv/bin/python workloads/research_probes/smsp_scheduler_contention/campaign.py manifest
.venv/bin/python workloads/research_probes/smsp_scheduler_contention/campaign.py correctness
.venv/bin/python workloads/research_probes/smsp_scheduler_contention/campaign.py ncu
.venv/bin/python workloads/research_probes/smsp_scheduler_contention/campaign.py source
.venv/bin/python workloads/research_probes/smsp_scheduler_contention/campaign.py nvbit
.venv/bin/python workloads/research_probes/smsp_scheduler_contention/campaign.py static
.venv/bin/python workloads/research_probes/smsp_scheduler_contention/campaign.py simulator
```
