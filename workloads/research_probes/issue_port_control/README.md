# sm_86 issue-port and control-pipeline probe

This probe separates dependent latency, independent initiation throughput, and
cross-family issue compatibility without memory, CTA barriers, asynchronous
copy, or atomics in the measured loop.

| Protocol | Construction |
|---|---|
| `p00` | loop/control baseline |
| `p10` | dependent carry-linked 64-bit add chain (`IADD3`/`IADD3.X`) |
| `p11` | dependent PTX `mad.lo.u32` chain (`IMAD` SASS) |
| `p12` | dependent PTX `lop3.b32` chain (`LOP3` SASS) |
| `p13` | uniformly taken predicate/branch pairs |
| `p20` | four independent carry-linked 64-bit add streams |
| `p21` | four-register independent `IMAD` streams |
| `p22` | four-register independent `LOP3` streams |
| `p31` | independent `IADD3`/FP32-FMA pairs |

Lengths 4, 8, 16, 32, and 64 are separate compile-time kernel
specializations. The long binary uses 8,192 loop generations and the NVBit
form uses 64. Every lane executes the chain; lane 0 of each warp reports
`clock64()` elapsed time and host-checked integer/FP32 results.

Build and run a smoke sample:

```bash
make -C workloads/research_probes/issue_port_control
workloads/research_probes/issue_port_control/issue_port_control p10 16 4 7
```

The campaign is frozen only after SASS inspection proves the requested
instruction families/counts and pilot NCU counters identify their physical
pipe attribution.

The frozen 68-cell matrix is resumable by stage:

```bash
.venv/bin/python workloads/research_probes/issue_port_control/campaign.py manifest
.venv/bin/python workloads/research_probes/issue_port_control/campaign.py correctness
.venv/bin/python workloads/research_probes/issue_port_control/campaign.py ncu
.venv/bin/python workloads/research_probes/issue_port_control/campaign.py source
.venv/bin/python workloads/research_probes/issue_port_control/campaign.py nvbit
.venv/bin/python workloads/research_probes/issue_port_control/campaign.py static
.venv/bin/python workloads/research_probes/issue_port_control/campaign.py simulator
.venv/bin/python workloads/research_probes/issue_port_control/analyze.py
```
