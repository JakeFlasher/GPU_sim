# SM86 DRAM demand/completion decomposition — H69 negative decision — 2026-10-16

## Independent real evidence

The retained RTX 3090 Ti probe campaigns were audited with their same-binary
NVBit/replay/NCU records, host identity (UUID
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, CC 8.6, driver 615.71.09, P0,
1350/10501 MHz), and crosswalk dispositions.

- `memory_retirement_service`: 102 real cells, all correctness/NVBit/replay/NCU
  complete; 94 instruction-crosswalk matches and 8 explicit system-fence
  crosswalk exclusions. Every retained cell is lower-bound consistent, but no
  hardware completion token or global event order is observed; dirty-writeback
  and coherence tails remain typed open.
- `store_ack_frontier`: 30 real arm cells, all crosswalk-valid and lower-bound
  consistent. Matched examples show non-universal store-tail deltas: at 84
  blocks/depth16/32 lanes, no-fence store is +3,500 GPC cycles over control and
  device-fence is +4,300; at depth64, the deltas are +11,400/+12,300; at
  one block, the same factors are roughly +0/+200–500. These are endpoint
  observations, not a completion timestamp or a transferable queue law.
- `launch_cycle_semantics`: a RED-tail control has a 2,245-cycle GPC gap
  (20.96%) over the simulator lower bound, with dirty/coherence/RMW tails open.
- Direct DRAM streams are mixed: the stride-mix cell is a lower-bound falsifier
  (simulator 550,005 cycles versus NCU median 419,040), while streaming dot
  product is lower-bound consistent (5,345 versus 7,562). Their demand, shape,
  and tail domains differ, so neither identifies a universal service law.

## Decision

No timing-active DRAM/writeback/coherence model is promoted. The evidence does
not identify whether the residual is read-service overlap, terminal dirty state,
coherence visibility, implicit EXIT drain, profiler launch envelope, or a
crosswalk/domain mismatch. In particular, terminal dirty residency is not
completed DRAM write traffic, and NCU GPC/SM-active values are not completion
witnesses.

The aggregate `DramBw::Carded` read service, typed dirty/coherence tails, and
observer-only demand candidates remain the default. H66 GCC/L1.5 demand rows
are source-domain observers and do not enter `DramStats.read_sectors` or timing.
A future timing arm requires a same-binary NVBit trace plus a locked one-factor
probe that isolates read demand, dirty eviction, coherence, and EXIT drain in
the same endpoint domain; otherwise retain interval/lower-bound reporting.

No full-corpus timing replay is claimed for H69 because no timing behavior was
changed. Existing H66 full-first evidence and raw/all-row H41 authority remain
valid.
