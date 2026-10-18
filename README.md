# Unified GPU simulator

A single Rust trace/replay engine for two explicit GPU deployment lanes:

| Lane | Architecture card | Capture/workload contract |
|---|---|---|
| RTX 3090 Ti | `arch_profiles/cards/sm_86.json` (`sm_86`) | SM86-native traces plus explicitly labeled portable controls; card-backed L1/L2/P-chase and DRAM hypotheses. |
| RTX 5090 | `arch_profiles/cards/sm_120-rtx-5090.json` (`sm_120`) | SM120/Blackwell traces with TMA/async sidecars and effective conditional L1/L2/IPC models; unmeasured geometry remains typed. |

```text
NVBit/SASS + sidecars -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run
NCU/CUDA observations ---------------------------------------> post-replay join
```

Architecture identity is selected from the card and validated against the
capture metadata. The two lanes share the trace, issue, memory and report
schemas while keeping architecture-specific hypotheses isolated. NCU values
are independent post-replay comparison targets, never replay/model inputs.

## Build and test

```bash
cargo check --manifest-path sim/Cargo.toml --workspace
cargo test  --manifest-path sim/Cargo.toml --workspace
cargo build --manifest-path sim/Cargo.toml --release -p sim-run
```

## Architecture-aware replay

`tools/sm120_corpus_replay.py` is retained as the historical entry point but
is architecture-neutral. Select a matching capture root and card:

```bash
# SM86 / RTX 3090 Ti
python3 tools/sm120_corpus_replay.py \
  --capture-root /path/to/sm86_rtx_3090_capture \
  --arch-card arch_profiles/cards/sm_86.json --target-arch sm_86 \
  --output /path/to/output --mode both --workers 12

# SM120 / RTX 5090
python3 tools/sm120_corpus_replay.py \
  --capture-root /path/to/sm120_rtx_5090_capture \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json --target-arch sm_120 \
  --output /path/to/output --mode both --workers 12
```

Use `--workers 1` only for an explicit sequential timing baseline. Full replay
writes compressed `.jsonl.zst`; fast replay is the separate trace-derived
frontier mode. SM86 full replay selects its P-chase card and bounded candidate
inputs; SM120 full replay selects effective L1/L2 and the explicitly marked
prototype carded-DRAM scenario. Use `--no-carded-dram` for the conservative
SM120 ablation.

## Repository map

* `sim/` — Rust trace frontend, scheduler, memory hierarchy, event engine and
  direct runner.
* `arch_profiles/` — architecture cards, schemas and typed SM86 candidates.
* `tools/nvbit_tracer/` — NVBit/SASS trace and sidecar producer.
* `tools/` — architecture-aware replay and post-replay comparison tools.
* `workloads/` — source workloads and explicit SM86 adaptation/control lanes.
* `docs/` — unified contract plus retained lane-specific evidence notes.
* `RESEARCH_COMPARISON.md` — exhaustive allowed-scope comparison of the two
  source branches and their capture/final-result interfaces.
* `VERIFICATION_REPORT.md` — representative real-trace baseline comparison.
