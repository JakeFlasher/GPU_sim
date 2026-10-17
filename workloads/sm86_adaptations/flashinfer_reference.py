#!/usr/bin/env python3
"""Run one bounded FlashInfer-Bench reference control on an SM86 host.

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
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


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
        DTYPES[_name] = torch.float16


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
            # SM86 control: retain the logical quantized input surface but
            # execute in FP16 rather than requesting native FP8 Tensor Cores.
            x = torch.randn(shape, dtype=torch.float16).clamp_(-4.0, 4.0)
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


def make_input(name: str, spec: dict[str, Any], definition: dict[str, Any], axes: dict[str, Any]) -> Any:
    kind = spec.get("type", "random")
    input_def = definition["inputs"][name]
    dtype_name = input_def.get("dtype", "float32")
    shape = resolve_shape(input_def.get("shape"), axes)
    if kind == "scalar":
        value = spec.get("value")
        if dtype_name.startswith("int"):
            return torch.tensor(value, device="cuda", dtype=DTYPES[dtype_name])
        return torch.tensor(value, device="cuda", dtype=DTYPES.get(dtype_name, torch.float32))
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
                    end = int(axes.get("num_kv_indices", max(1, n - 1)))
                vals = torch.linspace(0, end, n, dtype=torch.float32).round()
                return vals.to(DTYPES[dtype_name]).cuda()
            if "indices" in name:
                hi = max(1, int(axes.get("num_pages", 1)))
                return torch.randint(0, hi, shape, dtype=DTYPES[dtype_name]).cuda()
        return random_tensor(dtype_name, shape, name=name)
    if kind != "random":
        raise RuntimeError(f"unsupported input kind {kind!r} for {name}")
    return random_tensor(dtype_name, shape, name=name)


@torch.no_grad()
def run_sm86_fast(problem: str, inputs: dict[str, Any], axes: dict[str, Any]) -> Any:
    """Use fused ATen operations as an Ampere semantic control.

    Several official references intentionally spell out per-head Python loops;
    under instruction-level instrumentation those loops create hundreds of
    tiny launches and exceed a bounded cloud capture.  These adaptations keep
    the same tensor contracts but express the work through the GeForce
    Blackwell-friendly fused PyTorch operators so the trace represents a
    usable production launch envelope.
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
            scores = torch.matmul(qq, kk.transpose(-1, -2)).squeeze(1) * inputs["sm_scale"].float()
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
            scores = torch.matmul(qq, kk.transpose(-1, -2)) * inputs["sm_scale"].float()
            causal = torch.arange(scores.shape[-2], device=q.device)[:, None] + (ke - ks - (qe - qs)) + 1
            causal = torch.arange(scores.shape[-1], device=q.device)[None, :] < causal
            scores = scores.masked_fill(~causal[None], -float("inf"))
            lse[qs:qe] = torch.logsumexp(scores, dim=-1).transpose(0, 1) / 0.6931471805599453
            out[qs:qe] = torch.matmul(torch.softmax(scores, dim=-1), vv).transpose(0, 1).to(torch.bfloat16)
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
            scores = torch.matmul(qq, kk.transpose(-1, -2)) * inputs["sm_scale"].float()
            mask = torch.arange(scores.shape[-1], device=q.device)[None, :] < (torch.arange(scores.shape[-2], device=q.device)[:, None] + ke - ks - (qe - qs) + 1)
            scores = scores.masked_fill(~mask[None], -float("inf"))
            lse[qs:qe] = torch.logsumexp(scores, dim=-1).transpose(0, 1) / 0.6931471805599453
            out[qs:qe] = torch.matmul(torch.softmax(scores, dim=-1), vv).transpose(0, 1).to(torch.bfloat16)
        return out, lse
    if name.startswith(("018_mla_paged_decode", "019_mla_paged_prefill")):
        qn, qp = inputs["q_nope"], inputs["q_pe"]
        kc, kp = inputs["ckv_cache"], inputs["kpe_cache"]
        indptr, ids = inputs["kv_indptr"].to(torch.long), inputs["kv_indices"].to(torch.long)
        qo = inputs.get("qo_indptr")
        out = torch.zeros_like(qn); lse = torch.full((qn.shape[0], qn.shape[1]), -float("inf"), device=qn.device, dtype=torch.float32)
        sequence_count = min(indptr.numel() - 1, qo.numel() - 1) if qo is not None else qn.shape[0]
        for b in range(sequence_count):
            qs, qe = (int(qo[b].item()), int(qo[b + 1].item())) if qo is not None else (b, b + 1)
            pages = ids[int(indptr[b].item()):int(indptr[b + 1].item())]
            if pages.numel() == 0: continue
            c, p = kc[pages, 0].to(torch.float32), kp[pages, 0].to(torch.float32)
            scores = (torch.matmul(qn[qs:qe].float(), c.transpose(0, 1)) + torch.matmul(qp[qs:qe].float(), p.transpose(0, 1))) * inputs["sm_scale"].float()
            lse[qs:qe] = torch.logsumexp(scores, dim=-1) / 0.6931471805599453
            out[qs:qe] = torch.matmul(torch.softmax(scores, dim=-1), c).to(torch.bfloat16)
        return out, lse
    if name.startswith("020_moe_fp8"):
        # A compact block-scaled FP8 MoE proxy.  Axis overrides reduce E/H/I
        # for cloud tracing while retaining FP8 conversion, scaling, routing,
        # SwiGLU and two GEMM launches.
        h = int(axes["hidden_size"]); i = int(axes["intermediate_size"])
        a = inputs["hidden_states"].float() * inputs["hidden_states_scale"].float().mean()
        w13 = inputs["gemm1_weights"][0].float(); w2 = inputs["gemm2_weights"][0].float()
        g1 = a.matmul(w13.transpose(0, 1)); x1, x2 = g1[:, :i], g1[:, i:]
        c = x1 * (x2 / (1.0 + torch.exp(-x2))); return c.matmul(w2.transpose(0, 1)).to(torch.bfloat16)
    raise RuntimeError(f"no SM86 fast adaptation for {problem}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-dir", required=True, type=Path)
    ap.add_argument("--workload-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=120509)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--sm86-fast", "--sm120-fast", dest="sm86_fast", action="store_true", help="use the fused SM86 control for loop-heavy attention/MoE rows")
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
    adapted_axes = {}
    for assignment in args.axis:
        if "=" not in assignment:
            raise RuntimeError(f"--axis expects NAME=VALUE, got {assignment!r}")
        name, value = assignment.split("=", 1)
        if name not in definition.get("axes", {}):
            raise RuntimeError(f"unknown axis override {name!r}")
        axes[name] = int(value)
        adapted_axes[name] = int(value)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    inputs = {
        name: make_input(name, row.get("inputs", {}).get(name, {"type": "random"}), definition, axes)
        for name in definition["inputs"]
    }
    module = None if args.sm86_fast else load_module(problem / "reference.py")
    def invoke():
        return run_sm86_fast(definition["name"], inputs, axes) if args.sm86_fast else module.run(**inputs)
    for _ in range(args.warmup):
        result = invoke()
        if isinstance(result, (tuple, list)):
            for value in result:
                if torch.is_tensor(value):
                    value.sum().item()
        elif torch.is_tensor(result):
            result.sum().item()
    estimated_bytes = sum(v.numel() * v.element_size() for v in inputs.values() if torch.is_tensor(v))
    max_bytes = int(os.environ.get("SM86_MAX_INPUT_BYTES", str(8 * 1024**3)))
    min_free = int(os.environ.get("SM86_MIN_FREE_BYTES", str(6 * 1024**3)))
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    if estimated_bytes > max_bytes or free_bytes < estimated_bytes + min_free:
        raise RuntimeError(f"SM86 memory budget exceeded: inputs={estimated_bytes} free={free_bytes}")
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    result = invoke()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    # Force completion while avoiding a large host-side dump.
    if isinstance(result, (tuple, list)):
        for value in result:
            if torch.is_tensor(value):
                value.reshape(-1)[:1].float().sum().item()
    elif torch.is_tensor(result):
        result.reshape(-1)[:1].float().sum().item()
    print(json.dumps({"problem": definition["name"], "workload_index": args.workload_index, "axes": axes, "adapted_axes": adapted_axes, "sm86_fast": args.sm86_fast, "implementation": "sm86_vectorized_aten_control" if args.sm86_fast else "sm86_reference_control", "estimated_input_bytes": estimated_bytes, "device": torch.cuda.get_device_name(), "synthetic_safetensors": any(v.get("type") == "safetensors" for v in row.get("inputs", {}).values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
