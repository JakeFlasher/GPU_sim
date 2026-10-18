# Full replay runbook

Build from this checkout:

```bash
cargo build --release --manifest-path sim/Cargo.toml --workspace
```

The runner name is historical; `tools/sm120_corpus_replay.py` is
architecture-neutral. Full replay always uses `--workers 12` and writes
compressed JSONL. Use an output directory outside the capture tree.

## SM86 / RTX 3090 Ti

```bash
python3 tools/sm120_corpus_replay.py \
  --capture-root <sm86-capture-root> \
  --arch-card arch_profiles/cards/sm_86.json --target-arch sm_86 \
  --pchase-card data/pchase/sm_86_static_card.json \
  --output <output-dir> --mode full --memory-mode l1-l2 --workers 12
```

The complete SM86 corpus may contain expected `control_error` and
`replay_error` dispositions. The runner records them and returns success for
the documented SM86 status profile; use `--strict-ok` when every capture must
be `ok`.

## SM120 / RTX 5090

```bash
python3 tools/sm120_corpus_replay.py \
  --capture-root <sm120-capture-root> \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json --target-arch sm_120 \
  --output <output-dir> --mode full --memory-mode l1-l2 \
  --no-carded-dram --workers 12
```

Repeat with `--carded-dram` for the carded-DRAM scenario. SM120 full replay
resolves its effective L1/L2 defaults from the card/runner contract. Fast mode
is intentionally omitted here because merge correctness is established by the
full path.

## Comparison

The machine-readable checker accepts explicit baseline and unified paths:

```bash
python3 compare_numeric_full.py \
  --case sm120_disabled <baseline-disabled/full.jsonl.zst> <output-disabled/full.jsonl.zst> \
  --case sm120_enabled  <baseline-enabled/full.jsonl.zst>  <output-enabled/full.jsonl.zst> \
  --case sm86           <baseline-sm86/full.jsonl.zst>      <output-sm86/full.jsonl.zst>
```

It detects duplicate, missing, and extra capture IDs; checks status,
architecture, memory mode, cache scope, launch count, and return code; rejects
non-finite values; and compares common finite numeric fields while excluding
only documented wall-clock and additive observer fields. The historical SM120
`/` versus `__` capture-ID spelling is normalized explicitly and collisions are
reported as failures.
