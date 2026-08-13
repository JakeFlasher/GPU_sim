---
Use when: deciding whether `%clock64` or `%globaltimer` is appropriate for a non-root timing diagnostic; converting wall-clock time to approximate cycles under dynamic clocks; debugging SM-local cycle dispersion.
Primary raw source: PTX ISA special registers, `references/NVIDIA_Manuals/ptx_isa/10_special-registers.md`.
Supporting sources:
- CUDA Programming Guide `clock64()` documentation in `references/NVIDIA_Manuals/cuda_guide/`.
- `references/papers/knowledge_base/active/2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks.pdf`.
- `workloads/simbench_bw/nvml_covariate_join.py` and `workloads/simbench_bw/nvml_covariate_acceptance.toml`.
Repo touchpoints:
- `tools/nvbit_tracer/` - optional phase-boundary injection.
- `workloads/simbench_bw/` - current timing and covariate artifacts.
Historical comparison context: `measurement/pinned-protocol-v21.md` records one
fixed-clock sm_86 regime. It is optional outside reproduction of that dataset.
Version family: PTX ISA 9.x; sm_120 primary target, sm_86 local smoke target where applicable.
---

# `%clock64` and `%globaltimer` Under Dynamic Clocks

`%clock64` and `%globaltimer` are allowed diagnostic timing primitives, but they
do not replace the project's wall-clock target. CUDA-event and CUPTI Activity
elapsed time remain the primary observables.

## `%clock64`

- A 64-bit per-SM cycle counter.
- Absolute values are not comparable across SMs.
- Intra-SM deltas are meaningful when the measurement stays on the same SM.
- DVFS behavior is not specified by NVIDIA in a way that lets this project
  treat `%clock64 / wall_time` as a stable ground-truth frequency.

Use `%clock64` for short, structured diagnostics such as phase-boundary timing,
instruction-latency microprobes, or inter-SM dispersion fingerprints. Record
the measurement overhead and avoid treating it as an unperturbed wall-clock
measurement.

## `%globaltimer`

- A 64-bit global nanosecond timer.
- Documented as target-specific and intended for NVIDIA tools.
- Useful as a diagnostic wall-clock-like signal, but not a contractual
  replacement for CUDA events or CUPTI Activity timestamps.

## Dynamic Clocks

The historical v2.1 dev-host dataset used
`nvidia-smi -pm 1; -lgc 1350; -lmc 10501`; see
`measurement/pinned-protocol-v21.md`. That context makes its cycle conversion
more stable, subject to throttle events. For unlocked contexts, cycle
conversion is approximate and should be covariate-aware:

```text
cycles_estimate_per_repeat = elapsed_time_per_repeat * read_only_sm_clock_covariate
```

Read-only NVML or `nvidia-smi --query-*` clock samples are covariates. They do
not freeze frequency, and they can miss intra-kernel DVFS transitions.

## Practical Rules

- Use CUDA events or CUPTI Activity for primary elapsed time.
- Use `%clock64` only for intra-SM deltas or injected phase diagnostics.
- Use `%globaltimer` as a diagnostic cross-check, not as the only timing source.
- Record read-only clock, temperature, power, and throttle-state covariates when
  available.
- Clock locking and persistence mode can reduce variance where supported.
  Record the actual clock regime and test comparability before pooling labels
  from different regimes. See `measurement/pinned-protocol-v21.md` for one
  historical example.

## See Also

- `.claude/knowledge/measurement/pinned-protocol-v21.md`
- `.claude/knowledge/measurement/nonroot-cloud-contract.md`
- `.claude/knowledge/measurement/cupti-activity-api.md`
- `.claude/knowledge/measurement/nvbit-injection-playbook.md`
- `workloads/simbench_bw/README.md`

## Escalate to Raw When

- NVIDIA publishes a new statement on `%clock64`, `%globaltimer`, or DVFS
  semantics.
- A new architecture changes or removes either special register.
- Phase-injected measurements disagree sharply with CUDA-event or CUPTI Activity
  wall-clock measurements.
