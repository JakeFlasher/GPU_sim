"""Per-baseline SOLAR Stage 4 analysis builders.

Each landed pilot's reference algorithm has a unique compute / memory
fingerprint. This module owns the per-slug registry of analysis
builders that translate a ``(definition.json, workload_axes)`` pair
into the SOLAR ``total`` dict ``EinsumGraphPerfModel.predict`` consumes.

The builder for a slug returns either:

- a populated dict (the SOLAR ``analysis`` payload with ``total`` and
  ``metadata`` sub-dicts) — the harness will materialise it to a
  temporary ``analysis.yaml`` and call SOLAR Stage 4; or
- ``None`` — the harness honestly emits ``unknown_t_sol`` for this
  baseline rather than fabricating a roofline.

The rmsnorm-family builder covers four landed pilots:
``problem_001_fused_add_rmsnorm_h2048``,
``problem_002_fused_add_rmsnorm_h4096``,
``problem_024_rmsnorm_h2048``,
``problem_069_rms_norm``.

Dedicated builders cover the rest of the AC pilot set:
``problem_031_flux_timestep_guidance_projection_embedding`` (3
FP32 linears dominate),
``problem_014_fp8_yarn_rope_embedding`` (memory-bound mixed
int64 / BF16 with FP32 staging),
``problem_006_multimodal_rope_position_calculation`` (memory-bound
integer/indexing with no MACs),
``problem_020_nvfp4_linear_layer`` (FP4 dense GEMM at M*N*K MACs;
mixed BF16 / E2M1 / E4M3 / FP32 staging — precision flag routes
to ``MAC_per_cycle_nvfp4_tc``),
``problem_030_flux_concatenated_sequence_processing_with_split``
(one FP32 dense GEMM after a sequence-axis concat),
``problem_012_fused_cos_sin_embedding_generation`` (FP32 SFU
cos/sin + scale + BF16 cast; no MACs),
``problem_025_video_latent_gelu_activation`` (FP32 element-wise
GELU; no MACs).

Any other slug falls through to ``None`` until a bespoke builder
is written; the harness honestly reports ``unknown_t_sol`` for it.

Precision-key coverage table
---------------------------------------------------------------------

Each builder reports a dominant precision dispatch via
``_slug_precision()``. The dispatch maps to a MAC throughput key in
the resolved arch yaml (``MAC_per_cycle_<dispatch>_tc`` for tensor
core paths or ``MAC_per_cycle_fp32_sm`` for the FP32 SM-core path).
When a key is absent on the resolved arch yaml, SOLAR's perf model
has a 2-stage fallback chain that silently degrades to
``MAC_per_cycle_fp32_sm`` — producing a dishonest numeric
``t_sol_ms``. M3's precision guard (``predict_t_sol_ms`` arch-key
probe) closes that loop by refusing to call SOLAR when the
explicit token's MAC key is missing on the target arch.

The table below pins per-arch coverage. ``yes`` means the MAC key
is present in the arch yaml under
``SOLAR/configs/arch/<arch>.yaml`` and SOLAR can return a numeric
``runtime_ms``; ``no`` means the key is intentionally absent (the
hardware lacks that execution path) and the guard returns
``unknown_t_sol``.

============= ====== ====== ====== ====== ====== ====== ======
Precision     sm_86  sm_89  sm_90  sm_100 sm_100a sm_120 fallback
============= ====== ====== ====== ====== ====== ====== ======
bf16_tc       yes    yes    yes    yes    yes    yes    fp16_tc
fp16_tc       yes    yes    yes    yes    yes    yes    bf16_tc
tf32_tc       yes    yes    yes    yes    yes    yes    --
int8_tc       yes    yes    yes    yes    yes    yes    --
fp32_sm       yes    yes    yes    yes    yes    yes    --
fp32_tc       no     no     no     yes*   yes*   no     fp32_sm
fp8_tc        no     yes    yes    yes    yes    yes    --
nvfp4_tc      no     no     no     yes    yes    yes    --
mxfp8_tc      no     no     no     yes    yes    yes    --
============= ====== ====== ====== ====== ====== ====== ======

``yes*`` for ``fp32_tc`` on sm_100 / sm_100a reflects the
B200's selective FP32 tensor-core path; the in-tree arch yamls
currently model it via the FP32 SM fallback for simplicity.

The five tokens in the explicit allowed set
(``bf16_tc``, ``fp16_tc``, ``tf32_tc``, ``int8_tc``, ``fp32_sm``)
are the ones whose MAC keys are universally present across all
arch yamls authored in this repo (sm_86 / sm_90 / sm_100a /
sm_120). They are the safe set for builders to emit; tokens
outside this set will trip the M3 precision guard on an Ampere
host even when SOLAR's silent-fallback chain would otherwise have
returned a number.

For 3090ti.yaml specifically:
    MAC_per_cycle_fp32_sm  : 10752  (84 SMs * 128 FP32 ALU @ 1.89 GHz)
    MAC_per_cycle_int8_tc  : 42328  (160 TOPS dense INT8)
    MAC_per_cycle_tf32_tc  : 10582  (40 TFLOPS dense TF32)
    MAC_per_cycle_fp16_tc  : 21164  (80 TFLOPS dense FP16)
    MAC_per_cycle_bf16_tc  : 21164  (= FP16 on 3rd-gen TC)
    MAC_per_cycle_fp4_tc   : absent  (FP4 TCs are Blackwell-only)
    MAC_per_cycle_fp8_tc   : absent  (FP8 TCs are Hopper / Blackwell)
    MAC_per_cycle_fp32_tc  : absent  (no FP32 TC on Ampere)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import tempfile
import yaml  # type: ignore

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
_REPO_ROOT = _SCRIPTS_DIR.parent


# Explicit precision tokens emitted by builders. The harness's
# ``predict_t_sol_ms`` MUST refuse to call SOLAR when the token's
# MAC key is missing from the resolved arch yaml (closing SOLAR's
# silent FP32 fallback for unsupported precisions).
ALLOWED_PRECISION_TOKENS = frozenset({
    "bf16_tc",
    "fp16_tc",
    "tf32_tc",
    "int8_tc",
    "fp32_sm",
    # Tensor-core precisions that legitimately exist on some
    # arches; the guard's MAC-key probe controls availability.
    "nvfp4_tc",
    "fp8_tc",
    "fp32_tc",
    "mxfp8_tc",
})


def _resolved_local_arch_yaml() -> Optional[Path]:
    """Return the live host's SOLAR arch yaml, or ``None``.

    Lazy delegation to ``arch_resolver`` so test harnesses can
    monkeypatch the result.
    """
    try:
        from baselines_harness.arch_resolver import (
            resolve_local_arch_yaml,
        )
        from baselines_harness.surfaces.host_probe import probe_host
    except Exception:
        return None
    try:
        host_info = probe_host()
    except Exception:
        return None
    return resolve_local_arch_yaml(host_info)


def _arch_supports_precision(
    precision_token: str, arch_yaml: Path
) -> bool:
    """Probe the arch yaml for ``MAC_per_cycle_<token>``.

    Returns True when the key is present (numeric scoring is safe),
    False otherwise (the harness should emit ``unknown_t_sol``
    rather than trip SOLAR's silent FP32 fallback).
    """
    if precision_token not in ALLOWED_PRECISION_TOKENS:
        return False
    if not arch_yaml.is_file():
        return False
    try:
        arch = yaml.safe_load(arch_yaml.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(arch, dict):
        return False
    key = f"MAC_per_cycle_{precision_token}"
    return key in arch


def _solar_precision_for_token(token: str) -> Optional[str]:
    """Translate an explicit token (e.g. ``fp32_sm``) into the
    base precision string SOLAR's perf model accepts."""
    if token not in ALLOWED_PRECISION_TOKENS:
        return None
    if token.endswith("_tc"):
        return token[:-3]
    if token.endswith("_sm"):
        return token[:-3]
    return token


_PERF_MODEL_CACHE_KEY = "_solbench_solar_perf_model"


def _import_perf_model() -> Optional[Callable]:
    """Load SOLAR's ``EinsumGraphPerfModel`` without polluting
    ``sys.modules["solar"]``.

    ``solar/__init__.py`` transitively imports ``networkx`` via the
    einsum subpackage; the einsum stack is not part of this repo's
    runtime surface, so we bypass the package ``__init__`` and load
    only the three files the perf model needs
    (``common/constants.py``, ``common/utils.py``, ``perf/perf_model.py``)
    directly via ``importlib.util``.

    The files use absolute ``from solar.common.constants import …``
    imports, so during the load step they must resolve at their real
    qualified names in ``sys.modules``. To avoid permanently
    shadowing the real ``solar`` package, this function:

    1. Snapshots existing ``sys.modules`` entries at the ``solar.*``
       keys it is about to touch.
    2. Inserts stub / freshly-loaded modules at those keys long
       enough for the file-by-file load to resolve.
    3. Caches the loaded perf model under the private key
       ``_solbench_solar_perf_model`` so subsequent calls skip the
       load entirely.
    4. Restores the snapshot — original ``solar.*`` modules come
       back; if the keys were absent before, they are removed.

    The returned class object keeps strong references to the
    constants and utils it imported at load time, so calls into
    ``EinsumGraphPerfModel.predict`` continue to work after the
    snapshot is restored.
    """
    cached = sys.modules.get(_PERF_MODEL_CACHE_KEY)
    if cached is not None:
        return getattr(cached, "EinsumGraphPerfModel", None)

    import importlib.util
    import types

    solar_root = _REPO_ROOT / "SOLAR" / "solar"
    if not (solar_root / "perf" / "perf_model.py").is_file():
        return None

    touched_keys = (
        "solar",
        "solar.common",
        "solar.perf",
        "solar.common.constants",
        "solar.common.utils",
        "solar.perf.perf_model",
    )
    _MISSING = object()
    snapshot: Dict[str, Any] = {
        key: sys.modules.get(key, _MISSING) for key in touched_keys
    }

    try:
        for pkg in ("solar", "solar.common", "solar.perf"):
            stub = types.ModuleType(pkg)
            stub.__path__ = [
                str(solar_root / pkg.split(".", 1)[1])
                if "." in pkg
                else str(solar_root)
            ]
            sys.modules[pkg] = stub

        def _load(name: str, path: Path) -> object:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load spec for {name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        _load(
            "solar.common.constants",
            solar_root / "common" / "constants.py",
        )
        _load(
            "solar.common.utils",
            solar_root / "common" / "utils.py",
        )
        perf_mod = _load(
            "solar.perf.perf_model",
            solar_root / "perf" / "perf_model.py",
        )

        sys.modules[_PERF_MODEL_CACHE_KEY] = perf_mod
        return getattr(perf_mod, "EinsumGraphPerfModel", None)
    except Exception:
        sys.modules.pop(_PERF_MODEL_CACHE_KEY, None)
        return None
    finally:
        for key, original in snapshot.items():
            if original is _MISSING:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original


# ---------- per-slug analysis builders ----------------------------------


def _read_workload_axes(
    benchmark_root: Path,
    subset: str,
    benchmark_name: str,
    workload_index: int,
) -> Optional[Dict[str, Any]]:
    """Read the workload's ``axes`` dict for the requested index."""
    workload_path = benchmark_root / subset / benchmark_name / "workload.jsonl"
    if not workload_path.is_file():
        return None
    lines = [
        ln for ln in workload_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("//")
    ]
    if workload_index < 0 or workload_index >= len(lines):
        return None
    try:
        return json.loads(lines[workload_index]).get("axes")
    except json.JSONDecodeError:
        return None


def _rmsnorm_family_analysis(
    *,
    batch_total: int,
    hidden_size: int,
    has_residual: bool,
    bytes_per_element: int = 2,
) -> Dict[str, Any]:
    """Build the SOLAR ``total`` dict for the rmsnorm pattern.

    ``batch_total`` is the flattened batch dimension (B for 2D inputs,
    B*S for 3D). The kernel performs:

      1. (optional) residual add:    [B,H] = [B,H] + [B,H]
      2. sum-of-squares + mean:      sum_h x[b,h]² → [B,1] then /H + EPS
      3. rsqrt → inv_rms [B,1]
      4. rescale + weight multiply:  y[b,h] = (x[b,h]) * inv_rms[b] * weight[h]

    Total compute:
      - sum_of_squares: B*H MACs (square-and-add)
      - rescale: B*H multiplies (two muls per output, counted as 2*B*H)
      - rsqrt: B "other ops"
      - (residual add: B*H adds, counted as other-ops)

    External memory I/O (model_io):
      - read hidden_states [B,H], weight [H], (residual [B,H] if applicable)
      - write output [B,H]
    """
    bh = batch_total * hidden_size
    macs = bh                                  # sum-of-squares MACs
    flops = 2 * bh + bh + bh                   # square+sum, ×inv_rms, ×weight
    if has_residual:
        flops += bh                            # residual add
    other_ops = batch_total                    # rsqrt per row
    if has_residual:
        other_ops += bh                        # residual elementwise add (SM-side)

    weight_elems = hidden_size
    model_io_elems = (
        bh                                     # hidden_states read
        + weight_elems                         # weight read
        + bh                                   # output write
        + (bh if has_residual else 0)          # residual read (extra input)
    )
    intermediate_elems = 2 * batch_total       # sumsq + inv_rms scratch (B each)
    fused_elems = model_io_elems               # no intermediates in fused view
    fused_prefetched_elems = fused_elems
    unfused_elems = model_io_elems + 2 * intermediate_elems  # read/write of intermediates

    return {
        "metadata": {
            "bytes_per_element": bytes_per_element,
            "pattern": "rmsnorm",
            "has_residual": has_residual,
            "batch_total": batch_total,
            "hidden_size": hidden_size,
        },
        "total": {
            "macs": int(macs),
            "flops": int(flops),
            "other_ops": int(other_ops),
            "unfused_elements": int(unfused_elems),
            "fused_elements": int(fused_elems),
            "fused_prefetched_elements": int(fused_prefetched_elems),
            "model_io_elements": int(model_io_elems),
            "intermediate_elements": int(intermediate_elems),
            "weight_elements": int(weight_elems),
        },
    }


def _build_problem_001_fused_add_rmsnorm_h2048(axes: Dict[str, Any]) -> Optional[Dict]:
    bs = axes.get("batch_size")
    if not isinstance(bs, int):
        return None
    return _rmsnorm_family_analysis(
        batch_total=bs, hidden_size=2048, has_residual=True
    )


def _build_problem_002_fused_add_rmsnorm_h4096(axes: Dict[str, Any]) -> Optional[Dict]:
    bs = axes.get("batch_size")
    if not isinstance(bs, int):
        return None
    return _rmsnorm_family_analysis(
        batch_total=bs, hidden_size=4096, has_residual=True
    )


def _build_problem_024_rmsnorm_h2048(axes: Dict[str, Any]) -> Optional[Dict]:
    bs = axes.get("batch_size")
    if not isinstance(bs, int):
        return None
    return _rmsnorm_family_analysis(
        batch_total=bs, hidden_size=2048, has_residual=False
    )


def _build_problem_069_rms_norm(axes: Dict[str, Any]) -> Optional[Dict]:
    # Problem 069 is 3D `[batch_size, seq_len, hidden_size=8192]`; the
    # kernel flattens to one row per (batch_size * seq_len) entry. The
    # reference at sol-execbench/data/benchmark/L1/069_rms_norm/reference.py:6
    # computes ``x = residual + hidden_states`` BEFORE RMSNorm, and the
    # landed CUDA kernel at baselines/L1/problem_069_rms_norm/cuda_cpp/
    # kernel.cu reads both ``hidden_states`` AND ``residual`` and adds
    # them in FP32 (lines 63-69 in the sum-of-squares pass; lines 97-104
    # in the rescale pass). Despite the slug saying ``rms_norm`` rather
    # than ``fused_add_rmsnorm``, this workload IS the fused-add variant,
    # so the SOLAR perf model must include one full residual tensor
    # read and the corresponding per-element adds. Passing
    # ``has_residual=False`` here under-modeled bytes + flops and
    # gave every problem_069 row a slightly-wrong
    # ``local_sol_score``.
    bs = axes.get("batch_size")
    sl = axes.get("seq_len")
    if not isinstance(bs, int) or not isinstance(sl, int):
        return None
    return _rmsnorm_family_analysis(
        batch_total=bs * sl, hidden_size=8192, has_residual=True
    )


def _build_problem_031_flux_timestep_guidance_projection_embedding(
    axes: Dict[str, Any],
) -> Optional[Dict]:
    """FLUX combined timestep + text projection embedding (all FP32).

    Three dense linears dominate compute:
      L1: 768 -> 3072   (timestep MLP, weight 3072x768, bias 3072)
      L2: 3072 -> 3072  (timestep MLP, weight 3072x3072, bias 3072)
      L3: 768 -> 3072   (text embedder, weight 3072x768, bias 3072)

      MACs/batch = 768*3072 + 3072*3072 + 768*3072 = 14_155_776

    Pre-linear work is tiny: timestep scaling, outer product against
    half_time_embed_dim=384 frequencies, cos/sin, concat into a
    768-wide vector. Post-linear work is SiLU on a 3072-wide
    intermediate plus a final 3072-wide add of the text projection.
    """
    bs = axes.get("batch_size")
    if not isinstance(bs, int):
        return None

    inner_dim = 3072
    pooled_projection_dim = 768
    time_embed_dim = 768
    half_time_embed_dim = 384
    bpe = 4  # FP32 everywhere

    macs = bs * (
        inner_dim * time_embed_dim       # L1
        + inner_dim * inner_dim          # L2
        + inner_dim * pooled_projection_dim  # L3
    )
    flops = 2 * macs  # MAC = mul+add
    other_ops = (
        bs                                     # timestep *= 1000
        + bs * half_time_embed_dim             # outer product mul
        + 2 * bs * half_time_embed_dim         # cos + sin
        + 2 * bs * inner_dim                   # SiLU = sigmoid + mul
        + bs * inner_dim                       # final residual add
    )

    weight_elems = (
        inner_dim * time_embed_dim + inner_dim                # L1 W+B
        + inner_dim * inner_dim + inner_dim                   # L2 W+B
        + inner_dim * pooled_projection_dim + inner_dim       # L3 W+B
        + half_time_embed_dim                                 # freqs
    )
    weight_bytes = weight_elems * bpe
    activation_in_bytes = bpe * (bs + bs * pooled_projection_dim)    # timestep + pooled
    activation_out_bytes = bpe * bs * inner_dim                       # conditioning
    model_io_bytes = weight_bytes + activation_in_bytes + activation_out_bytes

    intermediate_elems = bs * (
        1                                # timestep_scaled
        + half_time_embed_dim            # args
        + half_time_embed_dim            # sin_embed
        + half_time_embed_dim            # cos_embed
        + time_embed_dim                 # timestep_embed (cat)
        + inner_dim                      # x after L1
        + inner_dim                      # x after SiLU
        + inner_dim                      # timestep_embed after L2
        + inner_dim                      # text_embed after L3
    )
    intermediate_bytes = intermediate_elems * bpe
    fused_bytes = model_io_bytes
    fused_prefetched_bytes = fused_bytes
    orojenesis_bytes = model_io_bytes + 2 * intermediate_bytes  # read+write of each

    return {
        "metadata": {
            "bytes_per_element": bpe,
            "pattern": "flux_timestep_projection",
            "batch_size": bs,
            "inner_dim": inner_dim,
            "pooled_projection_dim": pooled_projection_dim,
        },
        "total": {
            "macs": int(macs),
            "flops": int(flops),
            "other_ops": int(other_ops),
            "orojenesis_bytes": int(orojenesis_bytes),
            "fused_bytes": int(fused_bytes),
            "fused_prefetched_bytes": int(fused_prefetched_bytes),
            "model_io_bytes": int(model_io_bytes),
            "intermediate_bytes": int(intermediate_bytes),
            "weight_bytes": int(weight_bytes),
        },
    }


def _build_problem_014_fp8_yarn_rope_embedding(
    axes: Dict[str, Any],
) -> Optional[Dict]:
    """FP8-quantized YaRN rotary position embedding (memory-bound).

    Per row (one position):
      1. int64 -> fp32 cast for position_ids[s]
      2. 32 fp32 multiplies for outer(t, inv_freq) -> freqs[32]
      3. pad to 128 elems; per-row blockwise-1x128 amax reduce
      4. fp32 divide by scale, fp8 e4m3fn round, fp32 cast back,
         multiply by scale (the faithful FP8 round-trip)
      5. take freqs_final[32]; concat to emb[64]; sin/cos with mscale
      6. bf16 cast for both cos_emb and sin_emb outputs

    There are no MACs — the entire kernel is elementwise +
    transcendental + reduce; memory-bound on the BF16 output writes.
    """
    sl = axes.get("seq_len")
    if not isinstance(sl, int):
        return None

    dim = 64
    half_dim = 32
    padded_dim = 128

    # No matmul; mark MAC throughput unused.
    macs = 0
    # Elementwise / reduction counts.
    other_ops = (
        sl                                    # int64 -> fp32 cast
        + sl * half_dim                       # outer product
        + sl * padded_dim                     # amax reduce (1x128)
        + sl * padded_dim                     # divide by scale
        + sl * padded_dim                     # fp8 round
        + sl * padded_dim                     # fp32 cast back
        + sl * padded_dim                     # multiply by scale (inverse)
        + 2 * sl * dim                        # cos + sin
        + 2 * sl * dim                        # mscale multiply (cos+sin)
        + 2 * sl * dim                        # bf16 casts
    )
    flops = other_ops  # mostly fp32 elementwise

    # Memory (bytes; explicit because dtypes are mixed).
    input_bytes = 8 * sl + 4 + 4              # int64 ids + 2 fp32 scalars
    output_bytes = 2 * 2 * sl * dim            # bf16 cos + bf16 sin
    weight_bytes = 4 * half_dim                # fp32 inv_freq lookup (32 elems)
    model_io_bytes = input_bytes + output_bytes + weight_bytes

    # Per-op intermediates if unfused (fp32 mostly).
    intermediate_bytes = (
        4 * sl * half_dim          # freqs[s, 32] fp32
        + 4 * sl * padded_dim      # freqs_padded[s, 128] fp32
        + 4 * sl * padded_dim      # freqs_scaled fp32
        + 1 * sl * padded_dim      # freqs_fp8 e4m3
        + 4 * sl * padded_dim      # freqs_dequant fp32
        + 4 * sl * half_dim        # freqs_final[s, 32] fp32
        + 4 * sl * dim             # emb[s, 64] fp32
    )

    fused_bytes = model_io_bytes
    fused_prefetched_bytes = fused_bytes
    orojenesis_bytes = model_io_bytes + 2 * intermediate_bytes

    return {
        "metadata": {
            # bf16 outputs dominate the model-io footprint; the SOLAR
            # *_bytes path bypasses bpe for cycle math, but we record
            # the predominant element size for the report display.
            "bytes_per_element": 2,
            "pattern": "fp8_yarn_rope_embedding",
            "seq_len": sl,
            "qk_rope_head_dim": dim,
        },
        "total": {
            "macs": int(macs),
            "flops": int(flops),
            "other_ops": int(other_ops),
            "orojenesis_bytes": int(orojenesis_bytes),
            "fused_bytes": int(fused_bytes),
            "fused_prefetched_bytes": int(fused_prefetched_bytes),
            "model_io_bytes": int(model_io_bytes),
            "intermediate_bytes": int(intermediate_bytes),
            "weight_bytes": int(weight_bytes),
        },
    }


def _build_problem_006_multimodal_rope_position_calculation(
    axes: Dict[str, Any],
) -> Optional[Dict]:
    """Qwen2.5-VL multimodal rope 3D position calculation.

    Pure integer / indexing work — scan input_ids[B, S] (int64),
    detect vision events via vision_start_token_id, look up the
    matching grid_thw entry, emit per-batch text/3D-vision position
    spans into position_ids[3, B, S] (int64). The only fp32 input
    is second_per_grid_ts[N_vid] used to compute temporal indices
    via long-cast. No matrix multiplies; memory-bound on the int64
    output writes (24 B/elem * 3 * B * S).
    """
    bs_raw = axes.get("batch_size")
    sl_raw = axes.get("seq_len")
    n_img_raw = axes.get("total_num_images")
    n_vid_raw = axes.get("total_num_videos")
    if not (
        isinstance(bs_raw, int)
        and isinstance(sl_raw, int)
        and isinstance(n_img_raw, int)
        and isinstance(n_vid_raw, int)
    ):
        return None
    bs: int = bs_raw
    sl: int = sl_raw
    n_img: int = n_img_raw
    n_vid: int = n_vid_raw

    # No matmul anywhere.
    macs = 0
    # Per-token scan + emit dominates; vision events add small per-event work.
    other_ops = bs * sl + 4 * (n_img + n_vid)  # scan + per-event indexing
    flops = other_ops

    # Memory bytes (explicit because of int64-dominated mix + fp32 scalar input).
    input_bytes = (
        8 * bs * sl            # input_ids int64
        + 8 * bs * sl          # attention_mask int64
        + 24 * n_img           # image_grid_thw int64 [N_img, 3]
        + 24 * n_vid           # video_grid_thw int64 [N_vid, 3]
        + 4 * n_vid            # second_per_grid_ts fp32
    )
    output_bytes = (
        24 * bs * sl           # position_ids int64 [3, B, S]
        + 8 * bs               # mrope_position_deltas int64 [B, 1]
    )
    weight_bytes = 0           # no model weights — purely operand-driven
    model_io_bytes = input_bytes + output_bytes

    # Single-pass kernel: scan once, emit once. No persisted intermediates.
    intermediate_bytes = 0
    fused_bytes = model_io_bytes
    fused_prefetched_bytes = fused_bytes
    orojenesis_bytes = model_io_bytes

    return {
        "metadata": {
            "bytes_per_element": 8,
            "pattern": "multimodal_rope_position_calculation",
            "batch_size": bs,
            "seq_len": sl,
            "total_num_images": n_img,
            "total_num_videos": n_vid,
        },
        "total": {
            "macs": int(macs),
            "flops": int(flops),
            "other_ops": int(other_ops),
            "orojenesis_bytes": int(orojenesis_bytes),
            "fused_bytes": int(fused_bytes),
            "fused_prefetched_bytes": int(fused_prefetched_bytes),
            "model_io_bytes": int(model_io_bytes),
            "intermediate_bytes": int(intermediate_bytes),
            "weight_bytes": int(weight_bytes),
        },
    }


def _build_problem_020_nvfp4_linear_layer(
    axes: Dict[str, Any],
) -> Optional[Dict]:
    """NVFP4 dense linear layer (FP4 GEMM dominates; mixed-dtype I/O).

    Reference flow:
      1. Quantize BF16 ``x`` via ``BlockWiseScalerNVFP4`` (per-tensor
         FP32 global scale + per-16-element FP8 e4m3 block scales),
         emit packed E2M1 ``x_fp4`` and FP8 ``x_scales``.
      2. Compute ``y = x_fp4 @ w_fp4^T`` via FP4 tensor cores with
         per-K-block FP8 scale folded at the MMA boundary and FP32
         accumulation.
      3. Apply combined global decode scale; cast to BF16 output.

    Dimensions are fixed by the problem spec: ``K = N = 8192``, scale
    block size ``SF = 16`` so ``K_blocks = 512``. ``M = B * S``.

    Mixed-dtype memory accounting per BL-20260513-solar-mixed-dtype-bytes:
    use explicit ``*_bytes`` keys (not ``*_elements``) because the
    workload spans BF16 (x, output), packed E2M1 (FP4 weights/x), FP8
    (scales), and FP32 (intermediate staging) — a single
    ``bytes_per_element`` flag would mis-count.
    """
    bs_raw = axes.get("batch_size")
    sl_raw = axes.get("seq_len")
    if not (isinstance(bs_raw, int) and isinstance(sl_raw, int)):
        return None
    bs: int = bs_raw
    sl: int = sl_raw
    M: int = bs * sl
    # Fixed by the problem (definition.json axes).
    N = 8192
    K = 8192
    SF = 16
    K_blocks = K // SF  # 512

    # FP4 dense matmul dominates compute.
    macs = M * N * K
    flops = 2 * macs

    # Quantization + pack + scale-factor work (host quant prepass +
    # FP4 pack + scale remap into CUTLASS interleaved SFA layout).
    # M*K covers per-element abs / scale / clip / pack (each pass).
    # M*K_blocks covers per-block scale-factor derivation + remap.
    other_ops = M * K + M * K_blocks

    # Model I/O bytes (operand-driven, contractual):
    #   2*M*K        : BF16 ``x`` input
    #   (N*K)//2     : packed E2M1 weight (2 nibbles per byte)
    #   N*K_blocks   : FP8 e4m3 weight scales
    #   2*M*N        : BF16 output
    #   4            : FP32 weight_global_decode_scale scalar
    model_io_bytes = (
        2 * M * K
        + (N * K) // 2
        + N * K_blocks
        + 2 * M * N
        + 4
    )

    # Per-op intermediates if the input quantization pipeline ran
    # unfused (mirrors the host quant prepass in our baseline):
    #   4*M*K          : FP32 scaled_x staging
    #   (M*K)//2       : packed x_fp4 nibbles
    #   M*K_blocks     : FP8 x_scales
    #   8*M*K_blocks   : FP32 per-block decode + encode scale scratch
    intermediate_bytes = (
        4 * M * K
        + (M * K) // 2
        + M * K_blocks
        + 8 * M * K_blocks
    )

    orojenesis_bytes = model_io_bytes + 2 * intermediate_bytes
    # Fused pipeline persists only the packed x_fp4 + x_scales on top
    # of the model I/O; everything else is fused away into registers /
    # SMEM during the CUTLASS dispatch.
    fused_bytes = model_io_bytes + (M * K) // 2 + M * K_blocks
    fused_prefetched_bytes = fused_bytes
    weight_bytes = (N * K) // 2 + N * K_blocks  # packed FP4 + FP8 scales

    return {
        "metadata": {
            # BF16 dominates the model-io footprint; the *_bytes path
            # bypasses bpe for cycle math, but the report display uses
            # bytes_per_element=2.
            "bytes_per_element": 2,
            "pattern": "nvfp4_linear_layer",
            "batch_size": bs,
            "seq_len": sl,
            "M": M,
            "N": N,
            "K": K,
            "K_blocks": K_blocks,
        },
        "total": {
            "macs": int(macs),
            "flops": int(flops),
            "other_ops": int(other_ops),
            "orojenesis_bytes": int(orojenesis_bytes),
            "fused_bytes": int(fused_bytes),
            "fused_prefetched_bytes": int(fused_prefetched_bytes),
            "model_io_bytes": int(model_io_bytes),
            "intermediate_bytes": int(intermediate_bytes),
            "weight_bytes": int(weight_bytes),
        },
    }


def _build_problem_030_flux_concatenated_sequence_processing_with_split(
    axes: Dict[str, Any],
) -> Optional[Dict]:
    """FLUX dual-stream sequence projection (all FP32).

    The reference concatenates ``encoder_hidden_states`` (text) and
    ``hidden_states`` (image) along the sequence dim, applies a single
    dense linear projection against ``process_weight`` (`hidden_dim` x
    `hidden_dim`), then splits back into separate streams. From a
    roofline standpoint this is one big GEMM with shape
    ``[B, total_S, D] @ [D, D] -> [B, total_S, D]`` where
    ``total_S = text_seq_len + img_seq_len``. The concat / split are
    memory operations; they do not add MACs but they DO add an
    intermediate footprint when materialised.

    Precision dispatch: FP32 inputs / outputs / weight. On Ampere
    (3090ti.yaml) the perf model falls back to
    ``MAC_per_cycle_fp32_sm`` for FP32; no FP32 tensor-core path.
    """
    bs_raw = axes.get("batch_size")
    text_sl_raw = axes.get("text_seq_len")
    img_sl_raw = axes.get("img_seq_len")
    if not (
        isinstance(bs_raw, int)
        and isinstance(text_sl_raw, int)
        and isinstance(img_sl_raw, int)
    ):
        return None
    bs: int = bs_raw
    text_sl: int = text_sl_raw
    img_sl: int = img_sl_raw
    # hidden_dim is fixed at 3072 by the definition.
    hidden_dim = 3072
    bpe = 4  # FP32 everywhere

    total_sl = text_sl + img_sl

    # One large dense matmul: [B, total_S, D] @ [D, D]
    macs = bs * total_sl * hidden_dim * hidden_dim
    flops = 2 * macs

    # No SM-side elementwise math besides the concat/split memory
    # shuffle.
    other_ops = 0

    weight_bytes = hidden_dim * hidden_dim * bpe
    activation_in_bytes = bpe * bs * total_sl * hidden_dim
    activation_out_bytes = bpe * bs * total_sl * hidden_dim
    model_io_bytes = (
        weight_bytes + activation_in_bytes + activation_out_bytes
    )

    # Intermediate: the materialised concatenation
    # ``concatenated[B, total_S, D]`` lives in DRAM between the
    # concat step and the matmul step in the unfused reference.
    intermediate_bytes = bpe * bs * total_sl * hidden_dim
    orojenesis_bytes = model_io_bytes + 2 * intermediate_bytes
    fused_bytes = model_io_bytes
    fused_prefetched_bytes = fused_bytes

    return {
        "metadata": {
            "bytes_per_element": bpe,
            "pattern": "flux_concat_linear_split",
            "batch_size": bs,
            "text_seq_len": text_sl,
            "img_seq_len": img_sl,
            "total_seq_len": total_sl,
            "hidden_dim": hidden_dim,
        },
        "total": {
            "macs": int(macs),
            "flops": int(flops),
            "other_ops": int(other_ops),
            "orojenesis_bytes": int(orojenesis_bytes),
            "fused_bytes": int(fused_bytes),
            "fused_prefetched_bytes": int(fused_prefetched_bytes),
            "model_io_bytes": int(model_io_bytes),
            "intermediate_bytes": int(intermediate_bytes),
            "weight_bytes": int(weight_bytes),
        },
    }


def _build_problem_012_fused_cos_sin_embedding_generation(
    axes: Dict[str, Any],
) -> Optional[Dict]:
    """Fused cos/sin embedding generation from RoPE frequencies.

    Reference:
      emb = cat(freqs, freqs)                # [B, S, head_dim]
      cos = cos(emb) * attention_scaling     # FP32
      sin = sin(emb) * attention_scaling     # FP32
      cast to BF16 → cos, sin outputs

    Input ``freqs`` is FP32 ``[B, S, half_head_dim]``; outputs ``cos``,
    ``sin`` are BF16 ``[B, S, head_dim]`` where ``head_dim = 2 *
    half_head_dim``. The cat is a logical replication (each input
    element drives one low-half output and one high-half output).

    Compute is dominated by cos / sin transcendentals on the SM
    (SFU) plus two FP32 multiplies per output element. No MACs.
    Precision dispatch: FP32 SM-side.
    """
    bs_raw = axes.get("batch_size")
    sl_raw = axes.get("seq_len")
    if not (isinstance(bs_raw, int) and isinstance(sl_raw, int)):
        return None
    bs: int = bs_raw
    sl: int = sl_raw
    head_dim_raw = axes.get("head_dim", 128)
    if not isinstance(head_dim_raw, int):
        return None
    head_dim: int = head_dim_raw
    half_head_dim = head_dim // 2

    # Per-output element work: 1 cos OR sin (SFU) + 1 scale mul + 1
    # cast. Two output tensors (cos and sin) cover the head_dim
    # dimension. The reference reads each freqs element twice (once
    # for the low half of emb, once for the high half), but a fused
    # kernel can read each freqs element ONCE and reuse the value
    # across the cat boundary. We follow the fused path here.
    elements_per_output = bs * sl * head_dim
    other_ops = 2 * elements_per_output * (1 + 1 + 1)  # cos+sin paths: SFU + mul + cast

    # No matmul.
    macs = 0
    flops = 0

    bpe_in = 4   # FP32 freqs
    bpe_out = 2  # BF16 cos / sin

    weight_bytes = 0  # No weights; attention_scaling is a host scalar.
    activation_in_bytes = bpe_in * bs * sl * half_head_dim + 4  # freqs + FP32 scalar
    activation_out_bytes = 2 * bpe_out * bs * sl * head_dim  # cos + sin
    model_io_bytes = (
        activation_in_bytes + activation_out_bytes
    )

    # Intermediate footprint when the cat is materialised first:
    # ``emb`` is [B, S, head_dim] FP32. A fused kernel skips this.
    intermediate_bytes = 4 * bs * sl * head_dim
    orojenesis_bytes = model_io_bytes + 2 * intermediate_bytes
    fused_bytes = model_io_bytes
    fused_prefetched_bytes = fused_bytes

    return {
        "metadata": {
            "bytes_per_element": bpe_in,  # FP32 dominant for compute
            "pattern": "fused_cos_sin_emb",
            "batch_size": bs,
            "seq_len": sl,
            "head_dim": head_dim,
            "half_head_dim": half_head_dim,
        },
        "total": {
            "macs": int(macs),
            "flops": int(flops),
            "other_ops": int(other_ops),
            "orojenesis_bytes": int(orojenesis_bytes),
            "fused_bytes": int(fused_bytes),
            "fused_prefetched_bytes": int(fused_prefetched_bytes),
            "model_io_bytes": int(model_io_bytes),
            "intermediate_bytes": int(intermediate_bytes),
            "weight_bytes": int(weight_bytes),
        },
    }


def _build_problem_025_video_latent_gelu_activation(
    axes: Dict[str, Any],
) -> Optional[Dict]:
    """GELU (tanh approximation) on FP32 video latent tensors.

    Reference:
      output = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    All FP32 element-wise on ``x[B, S, I=4096]``. Per element the work
    is roughly: 2 muls (x^3), 1 FMA (x + coeff*x^3), 1 scale mul, 1
    tanh (SFU), 1 add (1+tanh), 1 mul (0.5*x), 1 mul (final). No
    MACs; precision dispatch is FP32 SM-side.
    """
    bs_raw = axes.get("batch_size")
    sl_raw = axes.get("seq_len")
    if not (isinstance(bs_raw, int) and isinstance(sl_raw, int)):
        return None
    bs: int = bs_raw
    sl: int = sl_raw
    inter_raw = axes.get("intermediate_size", 4096)
    if not isinstance(inter_raw, int):
        return None
    intermediate_size: int = inter_raw

    elements = bs * sl * intermediate_size
    bpe = 4  # FP32 inputs / outputs

    # Per-output op count (counted at issue rate, not throughput):
    # x*x*x : 2 muls; sqrt_2_over_pi*(x + coeff*x³): 1 mul + 1 fma;
    # tanh: 1 SFU; (1 + tanh): 1 add; 0.5*x: 1 mul; *(1+tanh): 1 mul.
    # Eight ops total per element.
    other_ops = 8 * elements
    macs = 0
    flops = 0

    weight_bytes = 0  # No weights.
    activation_in_bytes = bpe * elements
    activation_out_bytes = bpe * elements
    model_io_bytes = activation_in_bytes + activation_out_bytes

    # Pure elementwise, no materialised intermediate beyond the
    # compute registers. Set intermediate_bytes = 0; orojenesis ==
    # model_io_bytes for the elementwise case.
    intermediate_bytes = 0
    orojenesis_bytes = model_io_bytes
    fused_bytes = model_io_bytes
    fused_prefetched_bytes = fused_bytes

    return {
        "metadata": {
            "bytes_per_element": bpe,
            "pattern": "elementwise_gelu_tanh",
            "batch_size": bs,
            "seq_len": sl,
            "intermediate_size": intermediate_size,
        },
        "total": {
            "macs": int(macs),
            "flops": int(flops),
            "other_ops": int(other_ops),
            "orojenesis_bytes": int(orojenesis_bytes),
            "fused_bytes": int(fused_bytes),
            "fused_prefetched_bytes": int(fused_prefetched_bytes),
            "model_io_bytes": int(model_io_bytes),
            "intermediate_bytes": int(intermediate_bytes),
            "weight_bytes": int(weight_bytes),
        },
    }


_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Optional[Dict]]] = {
    "problem_001_fused_add_rmsnorm_h2048": _build_problem_001_fused_add_rmsnorm_h2048,
    "problem_002_fused_add_rmsnorm_h4096": _build_problem_002_fused_add_rmsnorm_h4096,
    "problem_024_rmsnorm_h2048": _build_problem_024_rmsnorm_h2048,
    "problem_069_rms_norm": _build_problem_069_rms_norm,
    "problem_031_flux_timestep_guidance_projection_embedding":
        _build_problem_031_flux_timestep_guidance_projection_embedding,
    "problem_014_fp8_yarn_rope_embedding": _build_problem_014_fp8_yarn_rope_embedding,
    "problem_006_multimodal_rope_position_calculation":
        _build_problem_006_multimodal_rope_position_calculation,
    "problem_020_nvfp4_linear_layer": _build_problem_020_nvfp4_linear_layer,
    "problem_030_flux_concatenated_sequence_processing_with_split":
        _build_problem_030_flux_concatenated_sequence_processing_with_split,
    "problem_012_fused_cos_sin_embedding_generation":
        _build_problem_012_fused_cos_sin_embedding_generation,
    "problem_025_video_latent_gelu_activation":
        _build_problem_025_video_latent_gelu_activation,
}


# Slugs whose dominant compute is FP32 on SM cores. The harness
# emits the explicit ``fp32_sm`` token so the precision guard probes
# ``MAC_per_cycle_fp32_sm`` (universal across all in-tree arch
# yamls) instead of relying on SOLAR's silent fallback chain.
_FP32_SM_SLUGS = frozenset({
    "problem_031_flux_timestep_guidance_projection_embedding",
    "problem_006_multimodal_rope_position_calculation",
    "problem_030_flux_concatenated_sequence_processing_with_split",
    "problem_012_fused_cos_sin_embedding_generation",
    "problem_025_video_latent_gelu_activation",
})


# Slugs whose dominant compute is the NVFP4 dense MMA via the
# block-scaled tensor-core path; the precision guard probes
# ``MAC_per_cycle_nvfp4_tc`` from the resolved arch yaml. Absent on
# 3090ti.yaml (Ampere has no FP4 TCs) so a 3090 Ti run reports
# ``unknown_t_sol`` for this slug — by design.
_NVFP4_TC_SLUGS = frozenset({
    "problem_020_nvfp4_linear_layer",
})


def _slug_precision(slug: str) -> str:
    """Return the explicit precision token for the slug's dominant
    dispatch.

    Tokens are the harness's vocabulary; they are translated to
    SOLAR's base precision via ``_solar_precision_for_token`` at the
    perf-model boundary. The MAC-key probe in
    ``predict_t_sol_ms`` uses the explicit token directly.

    Per the precision-key coverage table (module docstring), the
    allowed tokens are ``{bf16_tc, fp16_tc, tf32_tc, int8_tc,
    fp32_sm, nvfp4_tc, fp8_tc, fp32_tc, mxfp8_tc}``. Builders
    declared here must emit one of these literally; the registry
    audit in ``test_precision_guard.py`` fails otherwise.
    """
    if slug in _FP32_SM_SLUGS:
        return "fp32_sm"
    if slug in _NVFP4_TC_SLUGS:
        return "nvfp4_tc"
    # Default: BF16 inputs/outputs with FP32 intermediate
    # accumulation (rmsnorm family and FP8 yarn rope). The BF16 TC
    # path is universally present on the arch yamls authored here.
    return "bf16_tc"


def build_analysis_for_workload(
    slug: str,
    benchmark_root: Path,
    subset: str,
    benchmark_name: str,
    workload_index: int,
) -> Optional[Dict[str, Any]]:
    """Look up the slug's builder and return its SOLAR analysis dict."""
    builder = _REGISTRY.get(slug)
    if builder is None:
        return None
    axes = _read_workload_axes(
        benchmark_root, subset, benchmark_name, workload_index
    )
    if axes is None:
        return None
    return builder(axes)


def predict_t_sol_ms(
    analysis: Dict[str, Any],
    precision: str,
    arch_yaml: Optional[Path] = None,
) -> Optional[float]:
    """Materialise the analysis to a temp yaml and run SOLAR Stage 4.

    ``precision`` is the harness's explicit token (e.g. ``fp32_sm``,
    ``bf16_tc``, ``nvfp4_tc``). The function:

    1. Resolves the arch yaml via ``_resolved_local_arch_yaml()`` if
       ``arch_yaml`` is not supplied. Tests can inject a synthetic
       yaml directly via the parameter.
    2. Probes the resolved yaml for ``MAC_per_cycle_<token>``.
       Missing key → ``None`` (renders as ``unknown_t_sol``); no
       silent fallback to FP32.
    3. Translates the token to SOLAR's base precision via
       ``_solar_precision_for_token`` and calls
       ``EinsumGraphPerfModel.predict``.

    Returns the ``fused_prefetched.runtime_ms`` value (a roofline
    lower bound on Tk), or ``None`` when SOLAR is unavailable, the
    precision is unsupported on the resolved arch, or the perf
    model fails.
    """
    perf_model_cls = _import_perf_model()
    if perf_model_cls is None:
        return None

    resolved_yaml: Optional[Path] = (
        arch_yaml if arch_yaml is not None else _resolved_local_arch_yaml()
    )
    if resolved_yaml is None or not resolved_yaml.is_file():
        return None
    if not _arch_supports_precision(precision, resolved_yaml):
        return None

    solar_precision = _solar_precision_for_token(precision)
    if solar_precision is None:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        analysis_path = tmp_path / "analysis.yaml"
        analysis_path.write_text(yaml.safe_dump(analysis), encoding="utf-8")
        perf = perf_model_cls().predict(
            analysis_path,
            tmp_path,
            arch_config=str(resolved_yaml),
            precision=solar_precision,
            copy_analysis=False,
        )
    if not isinstance(perf, dict):
        return None
    runtime = perf.get("fused_prefetched", {}).get("runtime_ms")
    if isinstance(runtime, (int, float)) and runtime > 0:
        return float(runtime)
    return None


def t_sol_for_baseline_workload(
    slug: str,
    benchmark_root: Path,
    subset: str,
    benchmark_name: str,
    workload_index: int,
    arch_yaml: Optional[Path] = None,
) -> Optional[float]:
    """Convenience: build analysis + run SOLAR + return ``t_sol_ms``.

    ``arch_yaml`` overrides the resolver's auto-selection. Tests can
    pin the arch yaml directly; production callers leave it ``None``
    so the live host's CC drives resolution.
    """
    analysis = build_analysis_for_workload(
        slug, benchmark_root, subset, benchmark_name, workload_index
    )
    if analysis is None:
        return None
    return predict_t_sol_ms(
        analysis, _slug_precision(slug), arch_yaml=arch_yaml
    )
