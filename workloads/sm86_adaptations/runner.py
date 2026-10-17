#!/usr/bin/env python3
"""Run one bounded FlashInfer/SOL reference control on an SM86 host.

This is intentionally a small, deterministic runner for cloud collection.  It
does not invoke the SOL judge or build a second extension: the checked-in
reference function is executed directly so NVBit sees the real PyTorch/CUDA
launch sequence.  Workloads that refer to unavailable FlashInfer safetensors
receive shape-correct synthetic index/scalar inputs; their disposition is
written to the run directory by the surrounding capture script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import torch


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
for _name in ("float8_e4m3fn", "float8_e5m2"):
    _dtype = getattr(torch, _name, None)
    if _dtype is not None:
        # SM86 has no native FP8 Tensor Core path. Keep the logical input
        # surface but execute the control in FP16 unless explicitly overridden.
        DTYPES[_name] = _dtype if __import__('os').environ.get('SM86_NATIVE_FP8') == '1' else torch.float16


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("flashinfer_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_shape(shape: list[Any] | None, axes: dict[str, Any]) -> tuple[int, ...] | None:
    if shape is None:
        return None
    out = []
    for dim in shape:
        if isinstance(dim, str):
            if dim.lstrip("-").isdigit():
                out.append(int(dim))
                continue
            if dim not in axes:
                raise RuntimeError(f"workload does not provide axis {dim!r}")
            out.append(int(axes[dim]))
        else:
            out.append(int(dim))
    return tuple(out)


def random_tensor(dtype_name: str, shape: tuple[int, ...], *, name: str) -> torch.Tensor:
    dtype = DTYPES[dtype_name]
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype).cuda()
    if dtype.is_floating_point:
        # Generate on the host.  CUDA random generation would itself launch
        # large Philox kernels before the workload and can dominate NVBit
        # traces for tiny production kernels.
        if "float8" in dtype_name:
            x = torch.randn(shape, dtype=torch.float32).clamp_(-4.0, 4.0).to(dtype)
        else:
            x = torch.randn(shape, dtype=dtype)
        # Float8 conversion is supported by CUDA 13/PyTorch cu130, but random
        # values are formed in FP32 if the host lacks a native generator.
        if x.dtype != dtype:
            x = x.to(dtype=dtype)
        return x.cuda()
    if dtype == torch.uint8:
        return torch.randint(0, 255, shape, dtype=dtype).cuda()
    return torch.randint(-8, 8, shape, dtype=dtype).cuda()


def make_input(name: str, spec: dict[str, Any], definition: dict[str, Any], axes: dict[str, Any], adapted_axes: dict[str, int]) -> Any:
    kind = spec.get("type", "random")
    input_def = definition["inputs"][name]
    dtype_name = input_def.get("dtype", "float32")
    shape = resolve_shape(input_def.get("shape"), axes)
    if kind == "scalar":
        value = spec.get("value")
        # SOL shape-only operators pass dimensions as scalar arguments instead
        # of tensors. An axis override must reach those arguments too: otherwise
        # the recorded shape can say S=8 while the GPU actually processes S=128.
        axis_name = name.removesuffix("_scalar")
        if axis_name in adapted_axes:
            value = adapted_axes[axis_name]
        return int(value) if dtype_name.startswith("int") else float(value)
    if shape is None:
        raise RuntimeError(f"input {name} has no shape and is not scalar")
    if kind == "safetensors":
        # The cloud checkout intentionally does not carry the multi-GB
        # FlashInfer trace blobs.  Build valid, bounded substitutes for index
        # tensors; dense inputs remain random.  This preserves launch shapes
        # without pretending to reproduce the unavailable source values.
        if dtype_name.startswith("int"):
            if "indptr" in name:
                n = shape[0]
                if "qo" in name:
                    end = int(axes.get("total_q", max(1, n - 1)))
                else:
                    end = int(axes.get("num_kv_indices", axes.get("total_kv", max(1, n - 1))))
                vals = torch.linspace(0, end, n, dtype=torch.float32).round()
                return vals.to(DTYPES[dtype_name]).cuda()
            if "indices" in name:
                hi = max(1, int(axes.get("num_pages", 1)))
                return torch.randint(0, hi, shape, dtype=DTYPES[dtype_name]).cuda()
        return random_tensor(dtype_name, shape, name=name)
    if kind == "bounded_index":
        # Explicitly bounded scatter/gather indices.  The generic integer
        # fallback below intentionally uses a small signed range for control
        # workloads, but that is invalid for token/index_add inputs and can
        # trigger a device-side assert before NVBit has a chance to finish.
        upper_axis = spec.get("upper_axis")
        upper = spec.get("upper")
        if upper is None and upper_axis:
            upper = axes.get(upper_axis)
        if upper is None:
            raise RuntimeError(f"bounded_index input {name} needs upper or upper_axis")
        upper = int(upper)
        if upper <= 0:
            raise RuntimeError(f"bounded_index input {name} has non-positive upper bound {upper}")
        return torch.randint(0, upper, shape, dtype=DTYPES[dtype_name]).cuda()
    if kind != "random":
        raise RuntimeError(f"unsupported input kind {kind!r} for {name}")
    return random_tensor(dtype_name, shape, name=name)


@torch.no_grad()
def run_sm86_fast(problem: str, inputs: dict[str, Any], axes: dict[str, Any]) -> Any:
    """Vectorized ATen reference adaptations, NOT FlashInfer fused kernels.

    Several official references intentionally spell out per-head Python loops;
    under instruction-level instrumentation those loops create hundreds of
    tiny launches and exceed a bounded cloud capture.  These adaptations keep
    the same attention or MoE operation while batching head computations.
    These are reference controls. The FP8 MoE path dequantizes to FP32 for
    matmul; it is NOT evidence for native FP8/block-scaled MMA execution.
    """
    name = problem
    if name.startswith(("012_gqa_paged_decode", "013_gqa_paged_decode")):
        q, kc, vc = inputs["q"], inputs["k_cache"], inputs["v_cache"]
        indptr, indices = inputs["kv_indptr"].to(torch.long), inputs["kv_indices"].to(torch.long)
        bsz, hq, d = q.shape
        hk = kc.shape[2]
        out = torch.zeros_like(q)
        lse = torch.full((bsz, hq), -float("inf"), device=q.device, dtype=torch.float32)
        for b in range(bsz):
            ids = indices[int(indptr[b].item()):int(indptr[b + 1].item())]
            if ids.numel() == 0:
                continue
            kk = kc[ids, 0].to(torch.float32).transpose(0, 1).repeat_interleave(hq // hk, dim=0)
            vv = vc[ids, 0].to(torch.float32).transpose(0, 1).repeat_interleave(hq // hk, dim=0)
            qq = q[b].to(torch.float32).unsqueeze(1)
            scores = torch.matmul(qq, kk.transpose(-1, -2)).squeeze(1) * inputs["sm_scale"]
            lse[b] = torch.logsumexp(scores, dim=-1) / 0.6931471805599453
            attn = torch.softmax(scores, dim=-1)
            out[b] = torch.bmm(attn.unsqueeze(1), vv).squeeze(1).to(torch.bfloat16)
        return out, lse
    if name.startswith(("014_gqa_paged_prefill", "015_gqa_paged_prefill")):
        q, kc, vc = inputs["q"], inputs["k_cache"], inputs["v_cache"]
        qo, kv, ids = inputs["qo_indptr"].to(torch.long), inputs["kv_indptr"].to(torch.long), inputs["kv_indices"].to(torch.long)
        hq, hk = q.shape[1], kc.shape[2]
        out = torch.zeros_like(q)
        lse = torch.full((q.shape[0], hq), -float("inf"), device=q.device, dtype=torch.float32)
        for b in range(qo.numel() - 1):
            qs, qe = int(qo[b]), int(qo[b + 1]); ks, ke = int(kv[b]), int(kv[b + 1])
            if qe <= qs or ke <= ks:
                continue
            kk = kc[ids[ks:ke], 0].to(torch.float32).transpose(0, 1).repeat_interleave(hq // hk, dim=0)
            vv = vc[ids[ks:ke], 0].to(torch.float32).transpose(0, 1).repeat_interleave(hq // hk, dim=0)
            qq = q[qs:qe].to(torch.float32).transpose(0, 1)
            scores = torch.matmul(qq, kk.transpose(-1, -2)) * inputs["sm_scale"]
            causal = torch.arange(scores.shape[-2], device=q.device)[:, None] + (ke - ks - (qe - qs)) + 1
            causal = torch.arange(scores.shape[-1], device=q.device)[None, :] < causal
            scores = scores.masked_fill(~causal[None], -float("inf"))
            lse[qs:qe] = torch.logsumexp(scores, dim=-1).transpose(0, 1) / 0.6931471805599453
            out[qs:qe] = torch.matmul(torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0), vv).transpose(0, 1).to(torch.bfloat16)
        return out, lse
    if name.startswith(("016_gqa_ragged", "017_gqa_ragged")):
        q, k, v = inputs["q"], inputs["k"], inputs["v"]
        qo, kv = inputs["qo_indptr"].to(torch.long), inputs["kv_indptr"].to(torch.long)
        hq, hk = q.shape[1], k.shape[1]
        out = torch.zeros_like(q); lse = torch.full((q.shape[0], hq), -float("inf"), device=q.device, dtype=torch.float32)
        for b in range(qo.numel() - 1):
            qs, qe = int(qo[b]), int(qo[b + 1]); ks, ke = int(kv[b]), int(kv[b + 1])
            if qe <= qs or ke <= ks: continue
            kk = k[ks:ke].to(torch.float32).transpose(0, 1).repeat_interleave(hq // hk, dim=0)
            vv = v[ks:ke].to(torch.float32).transpose(0, 1).repeat_interleave(hq // hk, dim=0)
            qq = q[qs:qe].to(torch.float32).transpose(0, 1)
            scores = torch.matmul(qq, kk.transpose(-1, -2)) * inputs["sm_scale"]
            mask = torch.arange(scores.shape[-1], device=q.device)[None, :] < (torch.arange(scores.shape[-2], device=q.device)[:, None] + ke - ks - (qe - qs) + 1)
            scores = scores.masked_fill(~mask[None], -float("inf"))
            lse[qs:qe] = torch.logsumexp(scores, dim=-1).transpose(0, 1) / 0.6931471805599453
            out[qs:qe] = torch.matmul(torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0), vv).transpose(0, 1).to(torch.bfloat16)
        return out, lse
    if name.startswith(("018_mla_paged_decode", "019_mla_paged_prefill")):
        qn, qp = inputs["q_nope"], inputs["q_pe"]
        kc, kp = inputs["ckv_cache"], inputs["kpe_cache"]
        indptr, ids = inputs["kv_indptr"].to(torch.long), inputs["kv_indices"].to(torch.long)
        qo = inputs.get("qo_indptr")
        out = torch.zeros_like(qn)
        lse = torch.full((qn.shape[0], qn.shape[1]), -float("inf"), device=qn.device, dtype=torch.float32)
        for b in range(indptr.numel() - 1):
            qs, qe = (int(qo[b]), int(qo[b + 1])) if qo is not None else (b, b + 1)
            pages = ids[int(indptr[b]):int(indptr[b + 1])]
            if pages.numel() == 0 or qs == qe:
                continue
            c, p = kc[pages, 0].float(), kp[pages, 0].float()
            scores = (torch.matmul(qn[qs:qe].float(), c.T) + torch.matmul(qp[qs:qe].float(), p.T)) * inputs["sm_scale"]
            if qo is not None:
                # Bottom-right causal alignment, including cached prefix.
                visible = (torch.arange(pages.numel(), device=qn.device)[None, :] <=
                           torch.arange(qe - qs, device=qn.device)[:, None] + pages.numel() - (qe - qs))
                scores = scores.masked_fill(~visible[:, None, :], -float("inf"))
            lse[qs:qe] = torch.logsumexp(scores, dim=-1) / 0.6931471805599453
            weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            out[qs:qe] = torch.matmul(weights, c).to(torch.bfloat16)
        return out, lse
    if name.startswith("020_moe_fp8"):
        # Same DeepSeek group-limited top-k routing and local expert semantics
        # as the supplied reference, generalized only over tensor geometry.
        a8 = inputs["hidden_states"]
        e_local, twice_i, h = inputs["gemm1_weights"].shape
        i = twice_i // 2
        block = 128
        a = a8.float() * inputs["hidden_states_scale"].T.repeat_interleave(block, dim=1)
        s13 = inputs["gemm1_weights_scale"].repeat_interleave(block, 1).repeat_interleave(block, 2)
        s2 = inputs["gemm2_weights_scale"].repeat_interleave(block, 1).repeat_interleave(block, 2)
        w13 = inputs["gemm1_weights"].float() * s13
        w2 = inputs["gemm2_weights"].float() * s2
        logits = inputs["routing_logits"].float()
        t, e_global = logits.shape
        n_group = min(8, e_global)
        group_size = max(1, e_global // n_group)
        topk_group, top_k = min(4, n_group), min(8, e_global)
        scores = torch.sigmoid(logits)
        scores_biased = scores + inputs["routing_bias"].float()
        group_scores = scores_biased.reshape(t, n_group, group_size).topk(min(2, group_size), dim=-1, sorted=False).values.sum(-1)
        group_ids = group_scores.topk(topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_ids, 1.0)
        mask = group_mask.repeat_interleave(group_size, dim=1)
        expert_ids = scores_biased.masked_fill(mask == 0, -float("inf")).topk(top_k, dim=1, sorted=False).indices
        weights = scores * torch.zeros_like(scores).scatter_(1, expert_ids, 1.0)
        weights = weights / (weights.sum(1, keepdim=True) + 1e-20) * inputs["routed_scaling_factor"]
        output = torch.zeros((t, h), dtype=torch.float32, device=a.device)
        local_start = int(inputs["local_expert_offset"])
        for le in range(e_local):
            ge = local_start + le
            if not 0 <= ge < e_global:
                continue
            selected = torch.nonzero((expert_ids == ge).any(1), as_tuple=False).squeeze(1)
            if selected.numel() == 0:
                continue
            g1 = a.index_select(0, selected).matmul(w13[le].T)
            up, gate = g1[:, :i], g1[:, i:]
            activation = up * gate * torch.sigmoid(gate)
            expert_out = activation.matmul(w2[le].T)
            output.index_add_(0, selected, expert_out * weights.index_select(0, selected)[:, ge, None])
        return output.to(torch.bfloat16)
    raise RuntimeError(f"no SM86 fast adaptation for {problem}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-dir", required=True, type=Path)
    ap.add_argument("--reference-file", type=Path, help="tracked adaptation of the supplied reference")
    ap.add_argument("--workload-index", type=int, default=0)
    ap.add_argument("--local-expert-offset", type=int)
    ap.add_argument("--seed", type=int, default=120509)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--sm86-fast", "--sm120-fast", dest="sm86_fast", action="store_true", help="use vectorized ATen SM86 control for attention/MoE rows")
    ap.add_argument(
        "--axis",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a declared axis for a bounded SM86 adaptation",
    )
    args = ap.parse_args()
    problem = args.problem_dir.resolve()
    definition = json.loads((problem / "definition.json").read_text())
    rows = [json.loads(line) for line in (problem / "workload.jsonl").read_text().splitlines() if line.strip()]
    row = rows[args.workload_index]
    axes = dict(row.get("axes", {}))
    for axis_name, axis_spec in definition.get("axes", {}).items():
        if axis_spec.get("type") == "const" and axis_name not in axes:
            axes[axis_name] = axis_spec["value"]
    # Resolve the small arithmetic expressions used by SOL definitions (for
    # example half_head_dim = head_dim // 2) after constants are installed.
    for _ in range(len(definition.get("axes", {}))):
        changed = False
        for axis_name, axis_spec in definition.get("axes", {}).items():
            if axis_name in axes or axis_spec.get("type") != "expr":
                continue
            expr = axis_spec.get("expression", "")
            try:
                axes[axis_name] = int(eval(expr, {"__builtins__": {}}, dict(axes)))
                changed = True
            except (NameError, TypeError, ValueError, SyntaxError, ZeroDivisionError):
                pass
        if not changed:
            break
    adapted_axes = {}
    for assignment in args.axis:
        if "=" not in assignment:
            raise RuntimeError(f"--axis expects NAME=VALUE, got {assignment!r}")
        name, value = assignment.split("=", 1)
        if name not in definition.get("axes", {}):
            raise RuntimeError(f"unknown axis override {name!r}")
        axes[name] = int(value)
        adapted_axes[name] = int(value)
    # Recompute derived axes after overrides (e.g. proj_out_size follows an
    # overridden intermediate_size rather than the original constant).
    for axis_name, axis_spec in definition.get("axes", {}).items():
        if axis_spec.get("type") == "expr":
            axes[axis_name] = int(eval(axis_spec.get("expression", ""), {"__builtins__": {}}, dict(axes)))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    reference_path = args.reference_file.resolve() if args.reference_file else problem / "reference.py"
    module = None if args.sm86_fast else load_module(reference_path)
    has_custom_inputs = any(v.get("type") == "custom" for v in row.get("inputs", {}).values())
    if has_custom_inputs:
        if module is None:
            raise RuntimeError("custom workload inputs require an adaptation reference module")
        # SOL references use both get_inputs() and the older gen_inputs()
        # spelling.  Accept either entry point so bounded SM120 adapters can
        # preserve the supplied source generator without an artificial rewrite.
        input_factory = getattr(module, "get_inputs", None) or getattr(module, "gen_inputs", None)
        if input_factory is None:
            raise RuntimeError("custom workload inputs require reference.py::get_inputs or ::gen_inputs")
        # Custom generators run before profiler start; ACTIVE_FROM_START=0 keeps
        # setup/RNG kernels out of the workload trace.
        inputs = input_factory(axes, torch.device("cuda"))
    else:
        inputs = {
            name: make_input(name, row.get("inputs", {}).get(name, {"type": "random"}), definition, axes, adapted_axes)
            for name in definition["inputs"]
        }
    if args.local_expert_offset is not None:
        inputs["local_expert_offset"] = args.local_expert_offset
    # Persist the actual source and geometry alongside the trace. A dataset
    # directory may be ignored by Git and edited later; no hashes are needed
    # to retain the plain source used for this invocation.
    run_description = {
        "problem": definition["name"], "workload_index": args.workload_index,
        "axes": axes, "adapted_axes": adapted_axes, "seed": args.seed,
        "reference_file": str(reference_path),
        "implementation": ("sm86_vectorized_aten_control" if args.sm86_fast else
                           "sm86_adapted_reference" if args.reference_file else
                           "supplied_aten_reference"),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "native_backend": "Ampere_FP16_BF16_control" if args.sm86_fast else None,
        "input_tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                          for k, v in inputs.items() if torch.is_tensor(v)},
        "input_scalars": {k: v for k, v in inputs.items()
                          if isinstance(v, (int, float, bool, str))},
    }
    import os
    artifact_root = os.environ.get("SM86_RUN_ARTIFACT_DIR") or os.environ.get("TRACES_FOLDER")
    if artifact_root:
        artifacts = Path(artifact_root) / "workload_source"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "runner.py").write_text(Path(__file__).read_text())
        (artifacts / "reference.py").write_text(reference_path.read_text())
        (artifacts / "definition.json").write_text(json.dumps(definition, indent=2) + "\n")
        (artifacts / "run.json").write_text(json.dumps(run_description, indent=2) + "\n")
        # Save a native workload's compiled bridge and source alongside its
        # reference. These are collected materials, not hashes or run gates.
        for source in getattr(module, "NATIVE_SOURCE_FILES", ()):
            import shutil
            source = Path(source)
            native = artifacts / "native"
            native.mkdir(exist_ok=True)
            shutil.copy2(source, native / source.name)
    estimated_bytes = sum(v.numel() * v.element_size() for v in inputs.values() if torch.is_tensor(v))
    max_bytes = int(os.environ.get("SM86_MAX_INPUT_BYTES", str(8 * 1024**3)))
    min_free = int(os.environ.get("SM86_MIN_FREE_BYTES", str(6 * 1024**3)))
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    if estimated_bytes > max_bytes or free_bytes < estimated_bytes + min_free:
        raise RuntimeError(f"SM86 memory budget exceeded: inputs={estimated_bytes} free={free_bytes} max={max_bytes} reserve={min_free}")
    def invoke():
        return run_sm86_fast(definition["name"], inputs, axes) if args.sm86_fast else module.run(**inputs)
    for _ in range(args.warmup):
        result = invoke()
    torch.cuda.synchronize()
    # Both NVBit ACTIVE_FROM_START=0 and NCU --profile-from-start off use this
    # exact workload-only region. No RNG, warmup or checksum kernels enter it.
    torch.cuda.cudart().cudaProfilerStart()
    result = invoke()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    implementation = (
        "sm86_vectorized_aten_control" if args.sm86_fast else
        "sm86_adapted_reference" if args.reference_file else
        "supplied_aten_reference"
    )
    print(json.dumps({"problem": definition["name"], "workload_index": args.workload_index, "axes": axes, "adapted_axes": adapted_axes, "sm86_fast": args.sm86_fast, "implementation": implementation, "local_expert_offset": args.local_expert_offset, "device": torch.cuda.get_device_name(), "estimated_input_bytes": estimated_bytes, "synthetic_safetensors": any(v.get("type") == "safetensors" for v in row.get("inputs", {}).values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
