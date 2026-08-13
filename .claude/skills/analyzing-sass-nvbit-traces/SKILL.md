---
name: analyzing-sass-nvbit-traces
description: Decode SASS and operate the repository's NVBit trace toolchain. Use for SASS opcodes, cubin/fatbin/PTX inspection, NVBit injection, trace capture, instruction-mix analysis, architecture opcode provenance, or trace-replay debugging.
---

# Analyze SASS and NVBit traces

Use the SASS and NVBit toolchain directly. Root `AGENTS.md` governs workflow;
there is no H3 hold, temporary-copy requirement, artifact approval, or reviewer
handoff.

## Knowledge routes

- SASS tools and unknown opcodes:
  `.claude/knowledge/sass/sass-tooling.md`
- Per-architecture opcode history:
  `.claude/knowledge/sass/sass-instruction-reference.md`
- Architecture timeline:
  `.claude/knowledge/patents/sass-evolution-timeline.md`
- NVBit runtime details:
  `.claude/knowledge/measurement/nvbit-injection-playbook.md`
- Tracer source and commands:
  `tools/nvbit_tracer/README.md`

## Useful commands

```bash
cuobjdump --dump-sass kernel.cubin
cuobjdump --dump-elf-symbols kernel.cubin
nvdisasm -gi kernel.cubin
cu++filt _Z9my_kernelPf
nvprune --target-gpu sm_120 in.fatbin -o out.cubin

make -C tools/nvbit_tracer -j"$(nproc)"
make -C tools/nvbit_tracer test
```

Capture:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
OUT="$REPO_ROOT/results/nvbit_capture/manual"
mkdir -p "$OUT"
CUDA_INJECTION64_PATH="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$OUT" \
  ./my_app
```

Run NCU in a separate matched execution when ground-truth metrics are needed.

## Architecture hints

| Trace contains | Likely minimum architecture |
|---|---|
| `HMMA`, `BSSY`, `BSYNC` | Volta `sm_70` |
| `LDSM`, `R2UR`, `UIADD3` | Turing `sm_75` |
| `LDGSTS`, `DMMA` | Ampere `sm_80` |
| `UTMA*`, `WARPGROUP`, `*GMMA` | Hopper `sm_90` |
| `OMMA`, `QMMA`, `LDT`/`STT`, `UTCCP` | Blackwell `sm_100`/`sm_120` |

These are routing clues, not substitutes for inspecting the actual binary.

## Lockstep surfaces

When changing wire or classifier semantics, update the consumers that share
them:

- `sim/trace/src/lanes.rs` and
  `tools/nvbit_tracer/lane_reconstruction_golden.json`
- `sim/mem/src/cache_op.rs` and
  `tools/cache_predictor/l1_cache_op_classifier.py`
- `sim/mem/src/l2.rs` and
  `tools/cache_replay/sector_cache_replay.py`

Use focused parity tests to catch implementation drift. Test success is a
correctness result, not permission to continue.
