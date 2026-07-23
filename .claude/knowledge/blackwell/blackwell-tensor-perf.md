---
Use when: reading historical parent-repo tensor-core calibration notes, looking up prior-art tensor-core modeling references, or deciding what must be re-measured under the current non-root contract.
Primary raw source: historical parent-repo notes and `references/papers/knowledge_base/active/2025__accurate_models_of_nvidia_tensor_cores.pdf`.
Supporting sources:
- `references/papers/knowledge_base/active/2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks.pdf`
- `.claude/knowledge/ptx/ptx-isa-tensor-cores.md`
- `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md`
Repo touchpoints:
- `workloads/simbench_bw/` - current workload harness for non-root timing and feature extraction.
- `tools/nvbit_tracer/` - current dynamic SASS instrumentation path.
Out of scope: active NCU calibration, simulator tensor-path validation, and reuse of deleted D6 plans as project direction.
Version family: Historical SM120 tensor notes from the parent project; re-measure before making current quantitative claims.
---

# Blackwell Tensor-Core Performance Notes

> Current-scope note: this card preserves historical tensor-core calibration
> context. The parent project used NCU-derived evidence and simulator wiring;
> this fork must not use NCU, hardware counters, or simulator validation as an
> active method.

Use this card as a caution list and prior-art pointer, not as a current
measurement result. Any paper draft or model in this fork that needs Blackwell
tensor timing must either:

1. cite public literature,
2. derive a feature from static SASS/PTX,
3. measure elapsed behavior with allowed non-root signals, or
4. clearly mark the number as historical parent-repo evidence.

## Historical Claim Snapshot

The parent project carried a historical claim that selected SM120 QMMA FP4/FP8
microbenchmarks showed approximately 32-cycle latency and 32-cycle initiation
interval for specific dependent/saturation cases. That claim was tied to an
NCU-based calibration workflow and must not be treated as an admissible current
measurement in this fork.

Do not copy those numbers into current results without re-validating them under
the non-root contract or citing a public paper that supports the exact claim.

## Current Project Use

Tensor-core behavior can still enter the current project through allowed
features:

- PTX/SASS opcode identity and operand shape.
- Static instruction counts and tensor-op families from `cuobjdump`,
  `nvdisasm`, or NVBit traces.
- CUDA-event or CUPTI Activity elapsed time for kernels that vary tensor
  instruction mix.
- NVBit phase timing in matched A/B runs where overhead is reported.

## Modeling Context

- [Accurate Models of NVIDIA Tensor Cores](references/papers/knowledge_base/active/2025__accurate_models_of_nvidia_tensor_cores.pdf) - tensor-core modeling prior art through pre-Blackwell generations.
- [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](references/papers/knowledge_base/active/2025__dissecting_the_nvidia_blackwell_architecture_with_microbenchmarks.pdf) - public Blackwell microbenchmarking context.

## See Also

- `.claude/knowledge/measurement/nonroot-cloud-contract.md`
- `.claude/knowledge/measurement/nvbit-injection-playbook.md`
- `.claude/knowledge/ptx/ptx-isa-tensor-cores.md`
- `.claude/knowledge/ptx/ptx-isa-sm100-blackwell.md`
- `.claude/knowledge/sass/sass-instruction-reference.md`

## Escalate to Raw When

- A claim depends on a specific tensor latency, throughput, or initiation
  interval.
- A public Blackwell tensor-core paper or NVIDIA guide revision adds numbers
  that supersede this historical note.
