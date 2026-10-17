# Paper figures

`generate_figures.py` produces three paper-native vector figures:

* `novelty_stack.svg`: observed evidence → causal replay → independent NCU
  authority, with inherited mechanisms and the integration boundary explicit;
* `architecture_pipeline.svg`: concrete `sim-trace`/`sim-sm`/`sim-engine`/
  `sim-mem`/`sim-run` architecture and one-way NCU dataflow;
* `causal_timeline.svg`: issue→L1→L2→service→wake-up event ownership and
  separated frontiers;
* `simt_interleaving.svg`: CPU-style threads, SIMT lockstep, and the replay's
  one-host-actor-per-SM interleaving strategy with explicit tradeoffs;
* `sota_matrix.svg`: scope-aware comparison against modern simulator and
  analytical-model families; and
* `results_dashboard.svg`: thresholded architecture metrics plus campaign cost
  boundaries.

`workflow.svg` and `trace_frontier.svg` remain as historical source-art assets
from `/home/jakeshea/GPU_sims_notes/gpu_simulator_presentation/`; they are not
used by the current paper.  PDF conversions preserve arXiv/LaTeX compatibility,
while SVGs remain editable provenance.  Result values are in
`../data/metric_subset.csv` and `../data/runtime.csv` and are copied from the
H116 SM86 report and the correct 339-profile full-only SM120 reduction described
in `docs/thesis_gpu_simulator_integration_study.md`.
