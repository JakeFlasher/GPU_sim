---
name: analyzing-sass-nvbit-traces
description: Decodes SASS dumps and NVBit instrumentation traces, identifies per-arch SASS mnemonic provenance (Volta HMMA, Turing uniform U*, Ampere LDGSTS, Hopper WARPGROUP/GMMA/UTMA*, Blackwell OMMA/QMMA/LDT/STT/UTCCP), and runs the cuobjdump / nvdisasm / cu++filt / nvprune toolchain. Use when prompts mention SASS opcodes, .cubin / .fatbin / .ptx artifacts, NVBit injection, NVBit unknown-opcode handling, NVBit 1.8 sm_120 binary-version workaround, kernel decomposition, instruction-mix analysis, opcode-to-min-CC reverse lookup, BSSY/BSYNC convergence barriers, or trace-replay tooling. Always prefer the distilled SASS cards before raw references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md.
---

# Analyzing SASS and NVBit Traces

This skill owns SASS-layer questions and the NVBit-based trace pipeline.

## Cards

| Topic | Card |
|---|---|
| SASS tooling: `cuobjdump`, `nvdisasm`, `cu++filt`, `nvprune`, NVBit unknown-opcode workflow | `.claude/knowledge/sass/sass-tooling.md` |
| Per-arch SASS mnemonic provenance (which ops are Blackwell-new vs Hopper-era vs older) | `.claude/knowledge/sass/sass-instruction-reference.md` |
| SASS-evolution timeline (per-generation deltas, opcode→min-CC reverse lookup) | `.claude/knowledge/patents/sass-evolution-timeline.md` |
| NVBit injection playbook (LD_PRELOAD / CUDA_INJECTION64_PATH; NVBit 1.8 / sm_120 / driver 610.x current — NVBit 1.8 documented ceiling 575.xx, out-of-support but verified) | `.claude/knowledge/measurement/nvbit-injection-playbook.md` |

## Toolchain quick-reference

```bash
cuobjdump --dump-sass kernel.cubin           # emit SASS
cuobjdump --dump-elf-symbols kernel.cubin    # symbols
nvdisasm -gi kernel.cubin                    # graphical CFG
cu++filt _Z9my_kernelPf                      # demangle
nvprune --target-gpu sm_120 in.fatbin -o out.cubin   # strip non-target

# NVBit injection (no driver-root required)
REPO_ROOT="$(git rev-parse --show-toplevel)"
TRACE_ROOT="$REPO_ROOT/results/nvbit_capture_staging/manual_trace"
mkdir -p "$TRACE_ROOT"
LD_LIBRARY_PATH="$REPO_ROOT/.toolchain/pb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_INJECTION64_PATH="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$TRACE_ROOT" \
./my_app
```

## Per-architecture SASS deltas (from `sass-evolution-timeline.md`)

| Generation | Min CC | New SASS opcodes (representative) | Retired |
|---|---|---|---|
| Volta | sm_70 | `HMMA`, `IMMA`, `BSSY`, `BSYNC`, `WARPSYNC`, `DEPBAR`, `*32I` immediates | `SSY`, `SYNC`, `XMAD` |
| Turing | sm_75 | uniform `U*` family (`R2UR`, `S2UR`, `UIADD3`, `ULDC`, `USEL`, `USHF`, `VOTEU`), `LDSM`, `BMMA` | — |
| Ampere | sm_80 | `LDGSTS` (cp.async), `LDGDEPBAR`, `DMMA`, `F2IP`, `I2FP` | — |
| Hopper | sm_90 | `UTMA*` (TMA family), `UCGABAR_*`, `WARPGROUP`, `WARPGROUPSET`, `*GMMA`, `UBLK*`, `SYNCS`, `CGAERRBAR`, `PREEXIT` | — |
| Blackwell | sm_100 / sm_120 | `OMMA` (FP4), `QMMA` (FP8), `LDT`/`LDTM`/`STT`/`STTM` (TMEM), `UTCCP`, `UTCSHIFT`, `UTCBAR`, `UTCHMMA/UTCOMMA/UTCQMMA`, `UGETNEXTWORKID`, `UVIRTCOUNT` | `WARPGROUP`, `WARPGROUPSET`, parts of `*GMMA` |

Trace-fingerprint hints:

| Trace contains | Inferred minimum architecture |
|---|---|
| `XMAD` | pre-sm_70 |
| `HMMA`, `BSSY`, `BSYNC` (no `U*`) | sm_70 (Volta) |
| `LDSM`, `R2UR`, `UIADD3` | sm_75 (Turing) |
| `LDGSTS`, `DMMA` (no `UTMA*`) | sm_80–sm_89 (Ampere/Ada) |
| `UTMA*`, `WARPGROUP`, `*GMMA` | sm_90 (Hopper) |
| `OMMA`, `QMMA`, `LDT`/`STT`, `UTCCP` | sm_100 / sm_120 (Blackwell) |

## NVBit 1.8 specifics

- Provision the exact prefix, if absent, only with
  `bash tools/nvbit_tracer/pinned_build_env.sh --ensure`; ordinary sourcing is
  verify/export-only. While Task7/H3 is open, canonical Git worktrees may run
  `make -C tools/nvbit_tracer check-pinned-env` but must not build, clean, or
  regenerate. Build verification is limited to a non-Git temporary copy with
  `NVBIT_TEMPCOPY_BUILD_OK=1`. Python readers in
  `workloads/simbench_bw/_pb_generated` are tracked authorities: restore
  missing bytes from Git exactly and never generate replacements in the
  canonical checkout. The custom build pins g++-14, nvcc 13.2, and exact
  protobuf 34.1. CUDA 13.3 remains valid for target workloads.
- The eight SOL injector call sites are policy-quarantined until Stage 7; do
  not describe the runtime helpers as universally enforced across the repo.
- Inject with `CUDA_INJECTION64_PATH`; do not combine it with `LD_PRELOAD` (no
  driver-root required, unlike profiling-permission APIs).
- Has a device-capability binary-version fallback observed on both sm_86 and
  sm_120 — see `tools/nvbit_tracer/README.md`.
- Four upstream patches applied; concurrency model documented in same README.
- Trace-consumer lockstep: `sim/trace/src/lanes.rs` is an EXACT port of `import_trace_contract.py::_reconstruct_lane_vector` + the per-slot positional check; shared golden fixture `tools/nvbit_tracer/lane_reconstruction_golden.json` is consumed by BOTH test suites (`test_lane_reconstruction_golden.py` / `sim/trace/tests/lane_golden.rs`) — extend the fixture in the same commit as any decoder change.
- Cache-op lockstep: `sim/mem/src/cache_op.rs` ports the cache-op token rules (12-token vocabulary, adjacency-contiguous LDG modifier matching, bare-`LDG.E`→CA arch set) from `tools/cache_predictor/l1_cache_op_classifier.py`; structural parity test `tools/cache_predictor/test_cache_op_rust_parity.py` — any Python rule/vocabulary change (e.g. the queued sm_120 bare-`LDG.E` default) must land on both surfaces in the same commit.
- L2 cache-core parity: `sim/mem/src/l2.rs` ports the sector-cache core (`replay_access_stream` + `_install_way`) from `tools/cache_replay/sector_cache_replay.py`; parity corpus `tools/cache_replay/test_l2_rust_parity.py` ↔ `sim/mem/src/l2.rs::tests::python_parity_corpus_pins` — regenerate the corpus pins in the same commit as any cache-core change.

## When to escalate

- Unknown opcode in trace? Read `sass-tooling.md` first; fall through to `references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md` (per-arch catalog).
- Bug in the tracer? Read `tools/nvbit_tracer/README.md` then the patch list.

## Cross-references

- PTX → SASS lowering (which PTX produces which SASS) → `interpreting-ptx-isa` + compile a kernel with `nvcc -keep`.
- Why a particular opcode exists (patent-level reasoning) → `reasoning-about-nvidia-architecture-history`.
- Measurement protocol → `.claude/knowledge/measurement/pinned-protocol-v21.md` (the validating-nonroot skill is RETIRED archaeology).
- Compilation → `compiling-cuda-artifacts`.

## Hard rules

The SASS toolchain is read-only. NVBit injection works without driver-root. NCU is allowed (first-class ground-truth instrument under pinned protocol v2.1 — see `measurement/pinned-protocol-v21.md`), but NEVER co-run NCU and NVBit in one process; collect NCU labels in a separate matched run.

## Cite consulted files

End every reply with `## Knowledge Consulted`.
