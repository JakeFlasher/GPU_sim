# Host State & Migration Log — 3090 Ti dev host → 5060 laptop replay checklist

Running log of every privileged (sudo) host-state change and pinned-toolchain
fact on the sm_86 dev host, so the identical protocol can be replayed on the
5060 laptop (post-M2) and the cloud GB202 box (post-M2, D14). Append-only;
newest entries last. Companion: `plans/sm86_soundness_report.md` (M0),
`plans/phase1_full_tier_results.md` (Phase-1 evidence).

## Host identity (sm_86 reference host)

- GPU: NVIDIA GeForce RTX 3090 Ti (GA102, 84 SMs, 6 MiB L2), PCI 01:00.0
- Driver: 610.43.02 (labels before 2026-07-06 pchase legacy card: 595.71.05 — driver
  drift flagged in SALVAGE_MANIFEST S1)
- CUDA UMD: 13.3 system / **13.2 pinned toolkit for all builds**
  (`/usr/local/cuda-13.2/bin`; system nvcc 13.3 + protoc 35.1 BREAK the tracer —
  use `tools/nvbit_tracer/pinned_build_env.sh`, protoc 34.1 via `.toolchain/pb`)
- RmProfilingAdminOnly=0 (NCU counters + NCU clock-control work unprivileged;
  only nvidia-smi clock/persistence changes need root)

## Privileged commands run (sudo, user-executed, in order)

| date | command | effect | replay on 5060? |
|---|---|---|---|
| 2026-07-06 | `sudo nvidia-smi -pm 1` | persistence mode ON (survives until reboot/driver reload) | YES — first step |
| 2026-07-06 | `sudo nvidia-smi -lgc 1350` | SM/graphics clock locked to 1350 MHz (= sm_86 card's `sm_clock_ref_mhz`) | YES — pick the 5060 card's NVML reference point, NOT 1350 blindly |
| 2026-07-06 | `sudo nvidia-smi -lmc 10501` | memory clock locked to 10501 MHz (the single full-rate GDDR6X point; idle was 810) | YES — query `nvidia-smi -q -d SUPPORTED_CLOCKS` for the 5060's full-rate mem point first |
| (pending) | `sudo nvidia-smi -rgc && sudo nvidia-smi -rmc` | UNLOCK (reset) both — run when measurement campaigns are done | note: locks do NOT survive reboot |

## Measurement-protocol version history

- **v1** (legacy trees, driver 595): no explicit clock control; NCU default
  (base, pre-2026.1 semantics). Labels non-comparable to v2+.
- **v2** (2026-07-06, commits 255c0017..f1770947): host `-pm 1` + `-lgc 1350`,
  NCU `--clock-control none`, `--cache-control all`, `--replay-mode kernel`,
  sidecar `measurement_protocol` block records observed host state. Memory
  clock FREE-RUNNING (ramps 810→10501 under load; wobbles long low-utilization
  microbench kernels — cost the pchase carveout-100 point its confidence gate).
- **v2.1** (2026-07-06, after `-lmc 10501`) — THE reference protocol from here
  on. v2 + memory clock locked; fixed core:mem ratio 1350:10501.
  - pchase under v2.1: carveout-100 RESOLVED at 32768 (the core-only lock had
    lost it to L2-plateau wobble — mem lock fixed it, as predicted); L2 hit
    256→243.6, DRAM 436→422.6 cycles (tighter dispersions ±7.0/±0.04); L1/shared
    bit-identical 33.0016/23.0016 for a THIRD clock regime. L2-capacity knee
    now typed (L2:DRAM contrast 1.73x too flat for the step rule) — vendor
    6 MiB geometry unaffected.
  - v2-vs-v2.1 NCU label spot-check (rmsnorm+softmax, reference config):
    cycles shift −2.4/−3.0%, L2 hit rate +3pt abs (rmsnorm), uncore metrics
    2–9% — REAL protocol bias above the ~1% floor. Counts unchanged (exact).
    Consequence: v2 matrix keeps its determinism CLASSES + cross-protocol
    exhibits; reference-config MEANS re-captured under v2.1
    (`data/annex/repeatability_matrix_v21_ref_20260706/`); M1 label freeze is
    v2.1-native. NEVER mix v2 and v2.1 timing labels in one accuracy claim.

## 5060-laptop migration checklist (execute only after M2 green)

1. `sudo nvidia-smi -pm 1` (persistence).
2. `nvidia-smi -q -d SUPPORTED_CLOCKS` → pick the laptop GB206's reference
   SM point (record it in the sm_120-laptop ArchProfile
   `nvml_reference_operating_point` FIRST — do not reuse 1350) and its
   full-rate mem point; `sudo nvidia-smi -lgc <ref>` + `-lmc <mem>`.
   Laptop caveat: DVFS/thermal envelopes may override locks — verify with
   `clocks_throttle_reasons.active` == 0x1 (idle) / 0x0 (load) during runs;
   sidecars record throttle bitmask either way.
3. Rebuild ALL binaries with the pinned 13.2 toolkit (never system nvcc);
   regenerate protobuf bindings with pinned protoc 34.1 (`pinned_build_env.sh`).
4. Re-run in order: M0 smoke (build → NVBit trace → NCU T1/T2/T3 ×5 →
   admission) → pchase card (`run_pchase_card.py --arch sm_120`) → transcribe
   latencies (`transcribe_latencies_to_arch_card.py`) → 510-capture
   repeatability matrix (`repeatability_matrix_runner.py --metrics-from-registry`)
   → determinism classes (`determinism_class_report.py`). Expect the S7
   Blackwell `lts` pct overcount; the parser's raw-counter fallback handles it.
5. Control-bits sidecars: re-run `emit_control_bits_sidecar.py --arch sm_120`
   per binary (Blackwell control-bit encoding differences are typed —
   unknown_encoding exits 3; extend the extractor before trusting sidecars).
6. NVBit tracing: NVBit 1.8 sm_120 needs the binary-version workaround (see
   `.claude/skills/analyzing-sass-nvbit-traces`). NEVER co-run NVBit and NCU.

## Non-privileged host facts worth replaying

- venv: `.venv` python 3.12 (uv-managed); tests: `.venv/bin/python -m pytest`.
- `TRACES_FOLDER` env is honored by the tracer only when `USER_DEFINED_FOLDERS=1`
  — otherwise traces land in `./traces/` (move to annex).
- **TRAP: never `git commit` while a label capture campaign is running** —
  the driver stamps `source_commit` per repeat; a mid-sweep commit splits the
  cell's provenance and the aggregator fail-closes it
  (`label_provenance_duplicate`), forcing a recapture of that cell.
- **TRAP (cost one full 17-cell capture round): the tracer .so links the
  pinned protobuf runtime.** Without `LD_LIBRARY_PATH=<repo>/.toolchain/pb/lib`,
  `CUDA_INJECTION64_PATH` dlopen fails SILENTLY — app runs untraced, zero
  stderr, no banner. `roster_trace_capture.py` now exports it AND preflights
  `ldd tracer_tool.so` fail-closed (`tracer_so_deps_unresolved`). On the 5060:
  recreate `.toolchain/pb` first (host-local, never committed), and treat a
  missing NVBit banner as injection-failure, not as success.
- Annex (`data/annex/`) is gitignored machine-local evidence — copy separately
  when migrating (control_bits_sm86_20260706/, repeatability_matrix_*/).
- `.toolchain/` (mamba protoc env) is host-local, never committed; recreate via
  `pinned_build_env.sh` instructions. Protobuf C++ rule (web-verified,
  protobuf.dev): C++ requires an EXACT gencode==runtime version match (a
  `#error`/version-check enforces it) — stricter than "gencode <= runtime";
  regenerate bindings with the pinned protoc 34.1 whenever the runtime moves.
- **NVBit 1.8 support window (web-verified)**: released 2026-04-06; its README
  pins CUDA >= 12.0, driver <= 575.xx, SM 3.5-12.1. Our dev host runs driver
  610.43.02 (CUDA 13.3 era) — OFFICIALLY OUT of the documented window. It
  demonstrably works here (17/17 contract-valid traces, zero drops), but every
  new host/driver bump needs the M0 trace smoke re-run before trusting traces,
  and a future NVBit release should be adopted when available.
- **Review-gate lessons (2026-07-06 pre-Phase-2 review, commits that follow)**:
  (a) percent-scale means must never be paired with fraction-scale stddevs —
  the first noise-floor publication understated hit-rate CVs 100x; (b) noise
  floors computed from ADMITTED evidence are right-censored at the admission
  gate — say so wherever they are quoted; (c) the frozen bundle now VERIFIES
  each sidecar's measurement_protocol instead of stamping the protocol string
  on trust; (d) the host-state sample in sidecars is post-capture,
  GPU-index-selectable (LABEL_DRIVER_GPU_INDEX), uses the current
  clocks_event_reasons spelling with deprecated-alias fallback, and now
  records mem clock + power + temp (Rule 2 NVML covariates).
