# sm_86 fixed-latency completion-distance probe

This probe isolates the interaction between fixed-latency register reads and
producer-result completion. Each measured generation contains exactly sixteen
bare `IMAD` instructions and no memory, barrier, atomic, or branch instruction.
The same accumulator is consumed after 1, 2, 4, or 8 issued IMADs.

The source cubin is not used as a measurement treatment. A checked patcher
derives two semantically identical cubins from it:

- `split`: every hot IMAD reads two registers from its carried-addend bank and one
  from the other bank;
- `bank3`: every hot IMAD reads three distinct registers from its carried-addend
  bank.

All factor registers hold the same runtime-loaded value, so changing their
physical identities does not change the result. Both cubins have identical
PCs, opcodes, predicates, destinations, carried addend/RAW edges, control
fields, and zero reuse masks. Their only pairwise byte differences are the first two
source-register fields of the selected IMAD words.

Build and run a smoke test:

```bash
make -C workloads/research_probes/fixed_latency_completion
workloads/research_probes/fixed_latency_completion/fixed_latency_completion_driver \
  workloads/research_probes/fixed_latency_completion/fixed_latency_completion_split.cubin \
  4 split 4 8192 7
```

The 32-cell campaign covers RAW distances 1/2/4/8 at W4/W8/W12/W16 for both
matched variants. It is resumable by stage through `campaign.py`.
