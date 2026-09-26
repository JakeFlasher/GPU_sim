# Tail-scenario arithmetic — 2026-09-26

The [CSV](sm86_tail_sensitivity_20260926.csv) retains formula outputs from the
[eight-row development replay](sm86_holdout_replay_20260926.md). Writeback width
is bytes divided by card peak rate; coherence width assigns one cycle to each
cross-SM overlapping sector. Neither mechanism is scheduled. See
[diagnostic definitions](experimental_diagnostics.md).

**Interpretation corrected:** the 3.43% mean writeback width and 17.04% mean
coherence width relative to GPC frontier are arithmetic under unrelated assumed
coefficients. Overlap includes read/read sharing and excludes multiplicity of
ownership transitions. The larger GEMM coherence number does not establish a
coherence bottleneck, and the SiLU writeback number does not prove a hardware
optimization priority. The earlier “consequential architecture study” conclusion
is withdrawn. No novel mechanism or held-out design result follows.

To make an architecture decision, implement a specified queue/visibility or
routing mechanism, compare independent frontiers under a trace-valid
intervention, and evaluate with matching observables. The table is not such an
experiment; raw values are retained only as historical diagnostics.
