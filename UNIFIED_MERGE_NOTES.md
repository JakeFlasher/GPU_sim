# Unified merge notes

The merge combines the SM86 and SM120 simulator lanes into the `main` branch
without modifying the read-only source checkout. The source comparison is
summarized in [`RESEARCH_COMPARISON.md`](RESEARCH_COMPARISON.md); current
contracts and run commands are in `docs/`.

## Retained inputs

Only small real-trace smoke inputs are tracked under `verification_inputs/`.
The complete SM86/SM120 capture trees, baseline results, control sidecars, and
full replay outputs are external or ignored generated data. This distinction
is intentional: tracked smoke inputs support quick checks, while complete
corpus verification is run with explicit external paths.

## Verification contract

* `cargo test --manifest-path sim/Cargo.toml --workspace` must pass.
* Complete full replay uses `--workers 12` for both SM120 modes and SM86.
* `compare_numeric_full.py` must report matching row identity/status metadata
  and zero common numeric-field differences.
* SM86 and SM120 architecture-specific hypotheses must remain isolated.

The final campaign counts and the causes of previously observed discrepancies
are recorded in [`docs/RESULTS.md`](docs/RESULTS.md).
