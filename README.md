# Unified GPU simulator

This repository provides one architecture-aware Rust trace/replay engine for
RTX 3090 Ti (SM86) and RTX 5090 (SM120) workloads. The card and capture
metadata select the lane; no SM86 hypothesis is silently applied to SM120.

```text
NVBit/SASS + sidecars -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run
NCU/CUDA observations ---------------------------------------> post-replay join
```

## Build and test

```bash
cargo build --release --manifest-path sim/Cargo.toml --workspace
cargo test --manifest-path sim/Cargo.toml --workspace
```

## Run and compare

Use [`docs/REPLAY.md`](docs/REPLAY.md) for complete-corpus commands. Full
replay uses twelve simulator workers and compressed `.jsonl.zst` output. The
machine-readable comparison is `compare_numeric_full.py`; it checks row
identity, status/architecture/mode metadata, launch counts, return codes, and
common numeric replay fields.

The current campaign summary is in [`docs/RESULTS.md`](docs/RESULTS.md).
Architecture and scientific boundaries are in [`docs/CONTRACT.md`](docs/CONTRACT.md)
and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Source and session boundaries

`/home/jakeshea/5090_branch/` is a strictly read-only reference checkout.
Only the five permitted source subtrees were used for the merge research; no
paper or manuscript files are simulator inputs. See `SESSION_RULES.md` and
`LONG_TERM_NOTES.md` for the persistent session contract.
