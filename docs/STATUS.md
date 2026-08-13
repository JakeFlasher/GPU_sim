# Current status

The repository has functional Rust libraries for event scheduling, enhanced
NVBit trace decoding, warp/SM replay, a sectored memory hierarchy,
architecture-card loading, and several NCU-shaped metric outputs.

The main technical gaps are:

1. make `gpc__cycles_elapsed.max` an end-to-end prediction with documented
   aggregation semantics;
2. simplify the application boundary so a researcher can run a trace and
   architecture configuration directly;
3. collect a representative `sm_86` NCU/NVBit cohort with direct commands;
4. compare predicted and observed metrics with the small evaluator in `eval/`;
5. measure simulator event throughput and optimize the dominant paths;
6. add `sm_120` behavior only from architecture-appropriate evidence.

The previous repository state contained extensive process machinery around
capture and evaluation. That machinery has been removed from the active
workflow. Test failures and measurement variance remain technical findings to
fix or document; neither pauses unrelated work.

The SOL-ExecBench and historical campaign implementations remain present for
future opt-in use, including their environments, datasets, schemas, and legacy
artifact formats. They no longer define the default simulator workflow.
