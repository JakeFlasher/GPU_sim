# Cycle metric boundary

Updated 2026-09-23. This is a short engineering note, not a promotion gate.

## NCU observable

The metric registry names the NCU GPC elapsed-cycle observable as
gpc__cycles_elapsed.max. Nsight Compute defines the roll-up over the selected
unit instances and reports cycles in that unit's clock domain. The exact
simulator-to-NCU correspondence is an empirical question; the name alone does
not establish equivalence.

## Simulator diagnostics

For each launch, the replay exposes:

* diagnostic_last_issue_cycles;
* terminal_modeled_event_time; and
* a model-relative BindingCycles reducer over issue, terminal-event-plus-one,
  and modeled-completion frontiers.

The reducer is useful for comparing simulator revisions and for locating
missing modeled tails. It is not an NCU prediction until a same-launch
measurement establishes the mapping.

## Launch scope

The simulator keeps one diagnostic record per launch and aggregates the
declared simulator metric without summing independent launches into a
single hardware envelope. Hardware launch overlap, serialization, GPC
topology, and clock behavior must be measured with the final NCU campaign.

## Final comparison

For the final accuracy result, record the device identity, driver and NCU
version, clock policy, cache/replay mode, launch scope, matching NVBit trace,
and simulator command. Compare the same observable and scope. Keep issue,
event, completion, launch, and unmodeled-tail evidence separate in the output.

No RTL, Verilog, physical-design, silicon-internal, hash/seal, or
documentation-gate workflow is required by this note.
