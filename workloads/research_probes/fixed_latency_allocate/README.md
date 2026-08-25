# sm_86 fixed-latency Allocate probe

This initial probe generates SASS candidates for the Control/Allocate and
register-file milestone. It holds 16 bare `IMAD` instructions per generation
fixed while changing only which of six live source registers occupy the three
data-source positions.

| Protocol | Candidate construction |
|---|---|
| `r00` | loop/control baseline |
| `r10` | alternating parity triplets (`a,c,e` / `b,d,f`) |
| `r11` | adjacent-register triplets |
| `r12` | candidate triplets aligned with each destination's physical bank |
| `r20` | one stable triplet in fixed operand positions |
| `r21` | the same three registers rotated across operand positions |

The source-level labels are not bank or reuse claims. A cell is retained only
after `nvdisasm` plus the simulator's SASS operand projection proves its exact
physical registers, two-bank demand, reuse mask, bare-`IMAD` opcode, control
word, and spill-free resource use.

Build with `make -C workloads/research_probes/fixed_latency_allocate`, then run
`workloads/research_probes/fixed_latency_allocate/fixed_latency_allocate r10 8 7`.

The exact bank pair is produced by patching only the three source-register
fields of the 16 hot r11 IMAD words in the native long and trace cubins:

```bash
.venv/bin/python workloads/research_probes/fixed_latency_allocate/patch_sm86_imad_sources.py \
  workloads/research_probes/fixed_latency_allocate/fixed_latency_allocate.cubin \
  workloads/research_probes/fixed_latency_allocate/fixed_latency_allocate_same_bank.cubin \
  workloads/research_probes/fixed_latency_allocate/r11_same_bank_patch.json \
  --report results/captures/sm86_fixed_latency_allocate_20260817_v1/metadata/r11_same_bank_patch.json
```

The frozen 28-cell campaign is resumable by stage:

```bash
.venv/bin/python workloads/research_probes/fixed_latency_allocate/campaign.py manifest
.venv/bin/python workloads/research_probes/fixed_latency_allocate/campaign.py correctness
.venv/bin/python workloads/research_probes/fixed_latency_allocate/campaign.py ncu
.venv/bin/python workloads/research_probes/fixed_latency_allocate/campaign.py source
.venv/bin/python workloads/research_probes/fixed_latency_allocate/campaign.py nvbit
.venv/bin/python workloads/research_probes/fixed_latency_allocate/campaign.py static
.venv/bin/python workloads/research_probes/fixed_latency_allocate/campaign.py simulator
.venv/bin/python workloads/research_probes/fixed_latency_allocate/analyze.py
```
