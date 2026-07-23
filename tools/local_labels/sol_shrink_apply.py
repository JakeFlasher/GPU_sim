#!/usr/bin/env python3
"""Build and validate reviewed SOL reduced-scale proxy cohorts.

The native SOL workload JSONL is immutable input.  A proxy is planned only for
an exact source row whose bound native diagnostic is host-capacity OOM and the
family has a reviewed 3090-Ti policy.  Policies may reduce workload axes, an
immutable external definition constant, or both; iterations, tolerances, UUIDs,
dtypes, and every unlisted contract remain unchanged.  Each row is bound by the
tuple ``(slug, source_workload_uuid, variant_id)``, both definition identities,
exact native/reduced JSON-line hashes, and closed definition/axes deltas.
Native-passing rows remain explicit ``no_proxy_needed`` row plans.  Proxy
correctness is evaluated per variant and once more as a cell cohort.  Cycles have
``claim_scope=reduced_shape_only`` and ``native_predictive=false``.

For an OOM row, an unsupported family or incomplete model produces a typed
``needs_model`` row and blocks problem-level proxy admission.  An unrecognized
geometry in a reviewed family fails closed.  A passing or non-capacity-failed
row never enters proxy planning and cannot be relabeled.  No row is sent through
the former generic repeated-halving path.

Usage:
  .venv/bin/python tools/local_labels/sol_shrink_apply.py \
      --runnability /absolute/runnability-root/runnability.external.v2.json \
      --reduced-output-root /absolute/owned/empty/shrink-root \
      --out /absolute/owned/empty/shrink-root/shrink.external.v3.json \
      [--outer-timeout 2000]

Exit: 0 complete; 2 setup/IO/validation error.  Invoking this command performs
GPU correctness evaluation; importing and testing its pure planner does not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import sol_reconstitution as sr  # noqa: E402
from tools.local_labels import sol_shrink_calculator as sc  # noqa: E402


HISTORICAL_ARTIFACT_ID = "sm86_sol_shrink_provenance_v1"
ARTIFACT_ID = "sm86_sol_shrink_provenance_external_v3"
PROVENANCE_CONTRACT_VERSION = 3
CLAIM_SCOPE = "reduced_shape_only"
REDUCED_WORKLOAD_ROOT = (
    REPO_ROOT / "data" / "local_labels" / "campaign" / "sol_reduced_workloads")
PROXY_VARIANT_ROOT = REDUCED_WORKLOAD_ROOT / "geometry_proxy_variants"
EXTERNAL_OWNER_MARKER_NAME = ".sol-shrink-owner-v1.json"
EXTERNAL_OWNER_PURPOSE = "sm86_sol_geometry_proxy_campaign_inputs"

CLASS_SHAPE_ADJUSTED = sc.CLASS_SHAPE_ADJUSTED
CLASS_REDUCED = sc.CLASS_REDUCED
CLASS_UNAVAILABLE = sc.CLASS_UNAVAILABLE
CLASS_NATIVE_REPLAY = sc.CLASS_NATIVE_REPLAY
CLASS_CORRECTNESS_FAILED = "correctness_failed"
CLASS_CALIBRATION_INCONCLUSIVE = "calibration_inconclusive"
CLASS_NEEDS_MODEL = "needs_model"

CAPACITY_GATED_REDUCED_STATUSES = frozenset({
    sr.HOST_CAPACITY_OOM, "geometry_unavailable", "no_safe_axis",
})
REPLAY_MODE_CHECK = "not_supported_by_sol_execbench_cli"
ROW_SHA256_SCOPE = "exact_json_line_bytes_excluding_lf_only"
RUNNABILITY_PATH = (
    REPO_ROOT / "data/local_labels/campaign/sol_runnability_sm86.json")
PROVENANCE_SCHEMA = (
    REPO_ROOT
    / "workloads/simbench_bw/schemas/sol_shrink_provenance_external.v3.schema.json")
ARTIFACT_SCHEMA = (
    REPO_ROOT
    / "workloads/simbench_bw/schemas/sol_shrink_provenance_artifact.external.v3.schema.json")
DEFERRED_JSON = REPO_ROOT / "tools" / "sol_baselines_harness" / "deferred.json"

# Every entry below is a reviewed family-specific proxy contract.  The policy
# name is deliberately data, not executable code: it is included in the
# source-closure identity and dispatched by the closed implementation below.
# Workload iterations, tolerances, UUIDs, dtypes, tensor contracts, and every
# unlisted axis remain unchanged.  A definition target is emitted only as an
# immutable external artifact; the native benchmark definition is never edited.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "problem_001_attention_softmax_dropout_value_matmul_backward": {
        "family": "attention_rectangular_backward", "policy": "p001_budget"},
    "problem_013_fused_residual_rms_norm_backward": {
        "family": "residual_rmsnorm_backward", "policy": "p013_batch"},
    "problem_018_fused_rope_with_qk_norm_and_kv_cache_update": {
        "family": "rope_qknorm_kv_cache_update", "policy": "p018_batch"},
    "problem_026_video_patch_embedding_projection": {
        "family": "video_patch_projection", "policy": "p026_video_spatial"},
    "problem_044_moe_expert_computation": {
        "family": "moe_grouped_gemm", "policy": "p044_experts",
        "definition_axis": "num_experts", "definition_value": 32,
        "text_replacements": {
            "/reference": [
                ["gate_proj_weights: (256, 2048, 7168)",
                 "gate_proj_weights: (num_experts, 2048, 7168)"],
                ["up_proj_weights: (256, 2048, 7168)",
                 "up_proj_weights: (num_experts, 2048, 7168)"],
                ["down_proj_weights: (256, 7168, 2048)",
                 "down_proj_weights: (num_experts, 7168, 2048)"],
            ],
        }},
    "problem_060_fused_attention_qk_matmul_scale_mask_softmax_backward": {
        "family": "attention_s2_materializing_backward", "policy": "p060_sequence"},
    "problem_067_flash_attention_gqa_ultralong": {
        "family": "flash_attention_gqa", "policy": "p067_sequence"},
    "problem_076_batched_expert_forward": {
        "family": "moe_batched_expert_forward", "policy": "p076_experts",
        "definition_axis": "num_experts", "definition_value": 16},
    "problem_002_decoder_layer_full_block": {
        "family": "decoder_full_block", "policy": "p002_batch_sequence"},
    "problem_005_swiglu_mlp_backward": {
        "family": "swiglu_mlp_backward", "policy": "p005_token_extent"},
    "problem_012_moe_expert_batched_execution_with_capacity_factor": {
        "family": "moe_capacity_batched", "policy": "p012_capacity",
        "definition_axis": "num_experts", "definition_value": 40},
    "problem_024_moe_expert_parallel_execution": {
        "family": "moe_expert_parallel", "policy": "p024_expert_load",
        "definition_axis": "n_routed_experts", "definition_value": 64,
        "text_replacements": {"/description": [[
            "256 routed MoE experts", "64 routed MoE experts"]]}},
    "problem_025_moe_expert_parallel_execution_backward": {
        "family": "moe_expert_parallel_backward", "policy": "p025_timing_floor",
        "definition_axis": "n_routed_experts", "definition_value": 16},
    "problem_026_moe_expert_parallel_execution_with_weighted_aggregation": {
        "family": "moe_weighted_aggregation", "policy": "p026_capacity",
        "definition_axis": "num_experts", "definition_value": 4,
        "text_replacements": {"/description": [[
            "all 16 experts", "all 4 experts"]]}},
    "problem_028_gqa_rotary_attention_core_backward": {
        "family": "gqa_rotary_backward", "policy": "p028_batch"},
    "problem_029_moe_sparse_routing_and_dispatch": {
        "family": "moe_sparse_routing", "policy": "p029_token_extent"},
    "problem_034_vision_language_cross_attention_fusion": {
        "family": "vision_language_cross_attention", "policy": "p034_coupled"},
    "problem_047_moe_training_token_repeat_and_expert_computation": {
        "family": "moe_training_repeat", "policy": "p047_experts",
        "definition_axis": "num_experts", "definition_value": 32},
    "problem_055_audio_encoder_conv_positional_layer_stack": {
        "family": "audio_encoder_stack", "policy": "p055_batch"},
    "problem_059_decoder_layer_full_block": {
        "family": "decoder_full_block", "policy": "p059_wave_preserving"},
    "problem_065_sparse_expert_dispatch_and_combine": {
        "family": "moe_sparse_dispatch", "policy": "p065_experts",
        "definition_axis": "num_local_experts", "definition_value": 32},
    "problem_070_basic_transformer_block": {
        "family": "transformer_block", "policy": "p070_spatial_sequence"},
    "problem_077_sam_hq_vision_encoder_window_partition_attention": {
        "family": "sam_hq_window_attention", "policy": "p077_window_grid"},
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True,
                       sort_keys=True, separators=(",", ":")) + "\n").encode()


def _ordered_definition_json_bytes(value: dict[str, Any]) -> bytes:
    """Serialize a SOL definition without changing semantic member order.

    SOL-ExecBench binds ``definition.inputs`` insertion order to the positional
    ``run()`` signature.  Sorting a proxy definition is therefore not a
    presentation-only canonicalization: it can reorder arguments and make an
    otherwise valid definition fail before compilation.  The parsed native
    definition already preserves JSON object order, and proxy planning only
    replaces existing leaves, so retain that order while still emitting a
    deterministic compact UTF-8 representation.
    """
    return (
        json.dumps(
            value, allow_nan=False, ensure_ascii=True, sort_keys=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _row_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _source_head() -> str:
    """Return the actual worktree HEAD without inheriting redirected Git state."""
    return sr.source_head()


def validate_external_reduced_output_root(path: Path) -> dict[str, Any]:
    """Validate and fingerprint one pre-existing external campaign-input root.

    The lexical absolute path must already be its canonical path.  This rejects
    a symlink at the root or in any ancestor before the first owner marker,
    checkpoint, proxy row, or cohort byte is created.
    """
    return sr.validate_external_campaign_root(path)


def _external_output_path(
    root_identity: Mapping[str, Any], name: str,
) -> Path:
    return sr.external_child_path(root_identity, name)


def _open_bound_root(root_identity: Mapping[str, Any]) -> int:
    """Open the bound root without following its final component and rebind it."""
    return sr.open_external_root(root_identity)


def _read_bound_direct_child(
    root_identity: Mapping[str, Any], name: str, *, require_readonly: bool = True,
) -> bytes:
    return sr.read_external_child(
        root_identity, name, require_readonly=require_readonly
    )


def _publish_immutable_or_verify(
    root_identity: Mapping[str, Any], name: str, payload: bytes,
) -> Path:
    """Create immutable bytes, or verify an identical crash-resume remnant.

    No existing bytes are ever replaced.  Exact readonly bytes may be reused
    after an interruption between exclusive publication and checkpoint fsync.
    """
    return sr.publish_external_child(
        root_identity, name, payload, allow_identical_existing=True
    )


def _variant_id(*, slug: str, source_uuid: str, native_sha256: str,
                outcome: str, reduced_sha256: str | None,
                model_id: str | None, native_definition_sha256: str,
                reduced_definition_sha256: str | None) -> str:
    identity = {
        "model_id": model_id,
        "native_definition_sha256": native_definition_sha256,
        "native_row_sha256": native_sha256,
        "outcome": outcome,
        "reduced_definition_sha256": reduced_definition_sha256,
        "reduced_row_sha256": reduced_sha256,
        "slug": slug,
        "source_workload_uuid": source_uuid,
        "variant_contract_version": 2,
    }
    return _sha256(canonical_json_bytes(identity))


def _definition_constants(definition: dict[str, Any]) -> dict[str, int]:
    axes = definition.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("definition axes must be an object")
    constants: dict[str, int] = {}
    for name, contract in axes.items():
        if (isinstance(contract, dict) and contract.get("type") == "const"
                and isinstance(contract.get("value"), int)
                and not isinstance(contract.get("value"), bool)):
            constants[name] = contract["value"]
    return constants


def _replace_axes(native: dict[str, Any], updates: dict[str, int]) -> dict[str, Any]:
    reduced = dict(native)
    for name, value in updates.items():
        if name not in native:
            raise ValueError(f"proxy policy references missing workload axis: {name}")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"proxy policy produced invalid workload axis {name}={value!r}")
        reduced[name] = value
    return reduced


def _mapped_axes(
    native: dict[str, Any], keys: tuple[str, ...],
    mapping: dict[tuple[int, ...], tuple[int, ...]],
) -> dict[str, Any]:
    try:
        source = tuple(native[key] for key in keys)
    except KeyError as exc:
        raise ValueError(f"proxy policy missing workload axis: {exc.args[0]}") from exc
    if source not in mapping:
        raise ValueError(f"reviewed proxy policy has no mapping for {keys}={source}")
    return _replace_axes(native, dict(zip(keys, mapping[source], strict=True)))


def _policy_reduced_axes(slug: str, native: dict[str, Any]) -> dict[str, Any]:
    """Return the single conservative, reviewed 3090-Ti proxy geometry."""
    policy = MODEL_SPECS[slug]["policy"]
    if policy in {"p044_experts", "p076_experts", "p047_experts", "p065_experts"}:
        return dict(native)
    if policy == "p001_budget":
        return _mapped_axes(native, ("batch_size", "seq_len_q", "seq_len_kv"), {
            (4, 1024, 2048): (1, 1024, 2048),
            (32, 691, 773): (2, 691, 773),
            (32, 512, 512): (8, 512, 512),
            (1, 4096, 4096): (1, 1024, 1024),
        })
    if policy == "p013_batch":
        return _mapped_axes(native, ("batch_size", "seq_len"), {
            (64, 8192): (16, 8192)})
    if policy == "p018_batch":
        return _replace_axes(native, {"batch_size": 4})
    if policy == "p026_video_spatial":
        return _replace_axes(native, {"height": 128, "width": 128})
    if policy == "p060_sequence":
        return _mapped_axes(native, ("batch_size", "seq_len_q", "seq_len_k"), {
            (1, 4096, 4096): (1, 1024, 1024)})
    if policy == "p067_sequence":
        return _replace_axes(native, {"seq_len": 2048})
    if policy == "p002_batch_sequence":
        return _mapped_axes(native, ("batch_size", "seq_len"), {
            (2, 8192): (1, 4096)})
    if policy == "p005_token_extent":
        return _mapped_axes(native, ("batch_size", "seq_len"), {
            (32, 128): (16, 128), (16, 256): (4, 256),
            (1, 4096): (1, 1024), (4, 1024): (2, 1024),
            (8, 512): (4, 512), (16, 211): (4, 211),
            (64, 128): (16, 128), (4, 1571): (1, 1571),
            (2, 2048): (1, 2048),
        })
    if policy == "p012_capacity":
        tokens = native.get("num_tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 16:
            raise ValueError("p012 num_tokens is outside reviewed capacity policy")
        return _replace_axes(native, {"num_tokens": 4 * (tokens // 16)})
    if policy == "p024_expert_load":
        tokens = native.get("num_tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError("p024 num_tokens is invalid")
        return _replace_axes(native, {"num_tokens": (tokens * 64 + 128) // 256})
    if policy == "p025_timing_floor":
        tokens = native.get("num_tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError("p025 num_tokens is invalid")
        return _replace_axes(native, {"num_tokens": (tokens * 16 + 128) // 256})
    if policy == "p026_capacity":
        return _mapped_axes(native, ("batch_size", "seq_len"), {
            (32, 128): (8, 128), (2, 256): (1, 128),
            (2, 512): (1, 256), (2, 449): (1, 225),
            (4, 128): (1, 128), (1, 8192): (1, 2048),
            (4, 256): (1, 256), (1, 2048): (1, 512),
            (2, 2048): (1, 1024), (1, 512): (1, 128),
            (32, 256): (8, 256), (2, 1024): (1, 512),
            (16, 256): (4, 256), (4, 512): (1, 512),
            (16, 128): (4, 128), (8, 256): (2, 256),
        })
    if policy == "p028_batch":
        return _mapped_axes(native, ("batch_size", "seq_len"), {
            (64, 853): (4, 853)})
    if policy == "p029_token_extent":
        return _mapped_axes(native, ("batch_size", "seq_len"), {
            (1, 8192): (1, 4096), (64, 128): (16, 128),
            (32, 256): (8, 256),
        })
    if policy == "p034_coupled":
        return _mapped_axes(native,
            ("batch_size", "lang_seq_len", "vision_seq_len"),
            {(1, 8192, 12544): (16, 512, 784)})
    if policy == "p055_batch":
        return _replace_axes(native, {"batch_size": 1})
    if policy == "p059_wave_preserving":
        return _mapped_axes(native, ("batch_size", "seq_len"), {
            (1, 8192): (8, 1024)})
    if policy == "p070_spatial_sequence":
        return _mapped_axes(native,
            ("batch_size", "spatial_seq_len", "encoder_seq_len"),
            {(2, 4093, 77): (8, 511, 77)})
    if policy == "p077_window_grid":
        batch, height, width = (
            native.get("batch_size"), native.get("height"), native.get("width"))
        if (batch, height, width) in {(1, 613, 613), (1, 1024, 1024)}:
            return _replace_axes(native, {"height": 224, "width": 224})
        if (batch, height, width) == (8, 256, 256):
            return _replace_axes(native, {"batch_size": 1})
        raise ValueError("p077 row is outside reviewed window-grid policy")
    raise ValueError(f"unsupported reviewed proxy policy: {policy}")


def _definition_variant(
    definition: dict[str, Any], spec: dict[str, Any],
) -> tuple[bytes | None, list[dict[str, Any]]]:
    axis = spec.get("definition_axis")
    if axis is None:
        return None, []
    axes = definition.get("axes")
    contract = axes.get(axis) if isinstance(axes, dict) else None
    if (
        not isinstance(contract, dict)
        or contract.get("type") != "const"
        or not isinstance(contract.get("value"), int)
        or isinstance(contract.get("value"), bool)
    ):
        raise ValueError(f"definition constant is not a reviewed integer const: {axis}")
    native_value = contract["value"]
    reduced_value = spec["definition_value"]
    if reduced_value >= native_value:
        raise ValueError(f"definition proxy does not reduce {axis}")
    topk = _definition_constants(definition).get("num_experts_per_tok")
    if isinstance(topk, int) and reduced_value < topk:
        raise ValueError(f"definition proxy {axis} is below routing top-k")
    reduced = json.loads(json.dumps(definition))
    reduced["axes"][axis]["value"] = reduced_value
    delta = [{
        "json_pointer": f"/axes/{axis}/value",
        "from": native_value,
        "to": reduced_value,
    }]
    replacements = spec.get("text_replacements", {})
    if not isinstance(replacements, dict):
        raise ValueError("definition text replacement policy is not an object")
    for pointer in sorted(replacements):
        if pointer not in {"/description", "/reference"}:
            raise ValueError(f"definition text replacement pointer is forbidden: {pointer}")
        field = pointer[1:]
        original = reduced.get(field)
        if not isinstance(original, str):
            raise ValueError(f"definition text field is not a string: {pointer}")
        updated = original
        rows = replacements[pointer]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"definition text replacement list is empty: {pointer}")
        for pair in rows:
            if (
                not isinstance(pair, list) or len(pair) != 2
                or not all(isinstance(value, str) and value for value in pair)
                or updated.count(pair[0]) != 1
            ):
                raise ValueError(
                    f"definition text is outside reviewed replacement template: {pointer}"
                )
            updated = updated.replace(pair[0], pair[1])
        reduced[field] = updated
        delta.append({"json_pointer": pointer, "from": original, "to": updated})
    delta.sort(key=lambda row: row["json_pointer"])
    return _ordered_definition_json_bytes(reduced), delta


def _axes_delta(native: dict[str, Any], reduced: dict[str, Any]) -> list[dict[str, Any]]:
    if set(native) != set(reduced):
        raise ValueError("reduced axes changed the axis key set")
    return [
        {
            "axis": axis,
            "from": native[axis],
            "to": reduced[axis],
            "factor": (native[axis] / reduced[axis]
                       if isinstance(native[axis], (int, float))
                       and isinstance(reduced[axis], (int, float))
                       and reduced[axis] else None),
        }
        for axis in sorted(native) if native[axis] != reduced[axis]
    ]


def _base_row_plan(*, cell: dict[str, Any], source_index: int,
                   source_uuid: str, native_sha256: str,
                   native_axes: dict[str, Any], definition_ref: str,
                   definition_sha256: str, family: str | None,
                   model_id: str | None, planning_status: str,
                   reason: str | None) -> dict[str, Any]:
    variant_id = _variant_id(
        slug=cell["slug"], source_uuid=source_uuid,
        native_sha256=native_sha256, outcome=planning_status,
        reduced_sha256=None, model_id=model_id,
        native_definition_sha256=definition_sha256,
        reduced_definition_sha256=None)
    return {
        "axes_delta": [],
        "claim_scope": CLAIM_SCOPE,
        "correctness_result": "not_applicable",
        "definition_delta": [],
        "definition_ref": definition_ref,
        "definition_sha256": definition_sha256,
        "execution_evidence": None,
        "family": family,
        "geometry": None,
        "model": None,
        "native_axes": native_axes,
        "native_predictive": False,
        "native_row_sha256": native_sha256,
        "planning_reason": reason,
        "planning_status": planning_status,
        "reduced_axes": None,
        "reduced_definition_ref": None,
        "reduced_definition_sha256": None,
        "reduced_row_sha256": None,
        "reduced_run_detail": None,
        "reduced_run_status": None,
        "row_sha256_scope": ROW_SHA256_SCOPE,
        "slug": cell["slug"],
        "source_workload_index": source_index,
        "source_workload_uuid": source_uuid,
        "static_floor": None,
        "variant_id": variant_id,
    }


def _plan_row(*, cell: dict[str, Any], source_index: int,
              source_row: dict[str, Any], source_bytes: bytes,
              native_run_status: str,
              definition: dict[str, Any], definition_ref: str,
              definition_sha256: str,
              reduced_definition_ref: str | None,
              reduced_definition_sha256: str | None,
              definition_delta: list[dict[str, Any]]) -> tuple[
                  dict[str, Any], bytes | None]:
    slug = cell["slug"]
    source_uuid = source_row.get("uuid")
    axes = source_row.get("axes")
    if not isinstance(source_uuid, str) or not source_uuid:
        raise ValueError(f"{slug} workload row {source_index} lacks uuid")
    if not isinstance(axes, dict):
        raise ValueError(f"{slug} workload row {source_index} lacks axes")
    native_sha = _sha256(source_bytes)
    if native_run_status != sr.HOST_CAPACITY_OOM:
        planning_status = (
            "no_proxy_needed" if native_run_status == sr.RUNNABLE
            else "not_shrink_eligible"
        )
        return _base_row_plan(
            cell=cell, source_index=source_index, source_uuid=source_uuid,
            native_sha256=native_sha, native_axes=axes,
            definition_ref=definition_ref, definition_sha256=definition_sha256,
            family=None, model_id=None, planning_status=planning_status,
            reason=(
                "native_workload_passed"
                if native_run_status == sr.RUNNABLE else
                f"native_workload_not_host_capacity:{native_run_status}"
            )), None

    spec = MODEL_SPECS.get(slug)
    if spec is None:
        return _base_row_plan(
            cell=cell, source_index=source_index, source_uuid=source_uuid,
            native_sha256=native_sha, native_axes=axes,
            definition_ref=definition_ref, definition_sha256=definition_sha256,
            family=None, model_id=None, planning_status="needs_model",
            reason="family_model_not_registered"), None

    model_id = f"sol_reviewed_oom_proxy.v2:{spec['policy']}"
    reduced_axes = _policy_reduced_axes(slug, axes)
    delta = _axes_delta(axes, reduced_axes)
    if not delta and not definition_delta:
        raise ValueError(f"{slug}:{source_uuid} policy produced an unchanged proxy")
    reduced_row = dict(source_row)
    reduced_row["axes"] = reduced_axes
    reduced_bytes = source_bytes if not delta else _row_bytes(reduced_row)
    reduced_sha = _sha256(reduced_bytes)
    if reduced_sha == native_sha and reduced_definition_sha256 is None:
        raise ValueError(f"{slug}:{source_uuid} proxy bundle equals native bundle")
    variant_id = _variant_id(
        slug=slug, source_uuid=source_uuid, native_sha256=native_sha,
        outcome="proxy_planned", reduced_sha256=reduced_sha,
        model_id=model_id, native_definition_sha256=definition_sha256,
        reduced_definition_sha256=reduced_definition_sha256)
    plan = {
        "axes_delta": delta,
        "claim_scope": CLAIM_SCOPE,
        "correctness_result": "pending",
        "definition_delta": definition_delta,
        "definition_ref": definition_ref,
        "definition_sha256": definition_sha256,
        "execution_evidence": None,
        "family": spec["family"],
        "geometry": {
            "classification": "reviewed_reduced_proxy",
            "fit": "must_pass_live_3090ti_correctness_certificate",
            "regime_flags": (["expert_count"] if definition_delta else [])
                + (["workload_geometry"] if delta else []),
            "reviewed_policy": spec["policy"],
        },
        "model": {
            "model_id": model_id,
            "parameters": {
                "definition_delta": definition_delta,
                "native_axes": axes,
                "reduced_axes": reduced_axes,
            },
        },
        "native_axes": axes,
        "native_predictive": False,
        "native_row_sha256": native_sha,
        "planning_reason": None,
        "planning_status": "proxy_planned",
        "reduced_axes": reduced_axes,
        "reduced_definition_ref": reduced_definition_ref,
        "reduced_definition_sha256": reduced_definition_sha256,
        "reduced_row_sha256": reduced_sha,
        "reduced_run_detail": None,
        "reduced_run_status": None,
        "row_sha256_scope": ROW_SHA256_SCOPE,
        "slug": slug,
        "source_workload_index": source_index,
        "source_workload_uuid": source_uuid,
        "static_floor": None,
        "variant_id": variant_id,
    }
    return plan, reduced_bytes


def _load_source_rows(
    workload: Path,
) -> list[tuple[dict[str, Any], bytes, int]]:
    rows: list[tuple[dict[str, Any], bytes, int]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(workload.read_bytes().split(b"\n"), 1):
        if not raw.strip() or raw.lstrip().startswith(b"//"):
            continue
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid workload JSON {workload}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"workload row is not an object {workload}:{line_number}")
        uuid = row.get("uuid")
        if not isinstance(uuid, str) or not uuid or uuid in seen:
            raise ValueError(f"duplicate/invalid workload uuid {workload}:{line_number}")
        seen.add(uuid)
        # Bind the exact JSON-line bytes (excluding only the line terminator).
        # ``stripped`` is for recognition/parsing only; hashing it would silently
        # normalize whitespace and overstate the byte-level provenance claim.
        rows.append((row, raw, line_number))
    if not rows:
        raise ValueError(f"workload file has no effective rows: {workload}")
    return rows


def _classify_bound_native_diagnostic(diagnostic: dict[str, Any]) -> str:
    """Return the row-level native status from one exact bound diagnostic.

    The evaluator's surface statuses (for example ``RUNTIME_ERROR`` and
    ``INVALID_REFERENCE``) do not themselves distinguish OOM from correctness.
    Reuse the certified classifier over the retained full trace so the same OOM
    signatures and precedence apply at cell and row granularity.
    """
    if diagnostic.get("diagnostic_present") is not True:
        raise ValueError("native workload diagnostic is not present")
    status = diagnostic.get("status")
    trace = diagnostic.get("trace")
    if not isinstance(status, str) or not status or not isinstance(trace, dict):
        raise ValueError("native workload diagnostic lacks status/trace")
    try:
        trace_text = json.dumps(
            trace, allow_nan=False, ensure_ascii=True, sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("native workload diagnostic trace is not JSON-safe") from exc
    parsed = sr._parse_traces(trace_text)
    if (
        len(parsed) != 1
        or sr._lower(parsed[0].get("status")) != sr._lower(status)
    ):
        raise ValueError("native workload diagnostic status/trace mismatch")
    row_status, _ = sr.classify_cli_outcome(
        returncode=0, stdout=trace_text, stderr="", expected_workloads=1,
    )
    return row_status


def _bind_native_diagnostics(
    cell: dict[str, Any],
    source_rows: list[tuple[dict[str, Any], bytes, int]],
) -> list[str]:
    """Bind one diagnostic bijectively to every immutable source JSONL row."""
    if cell.get("status") != sr.HOST_CAPACITY_OOM:
        raise ValueError("shrink planner accepts only host-capacity cells")
    diagnostics = cell.get("workload_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != len(source_rows):
        raise ValueError("native workload diagnostic cardinality mismatch")
    if (
        cell.get("diagnostics_complete") is not True
        or cell.get("unexpected_workload_diagnostics") != []
    ):
        raise ValueError("native workload diagnostics are not complete/bijective")

    statuses: list[str] = []
    for source_index, ((source_row, source_bytes, source_line), diagnostic) in enumerate(
        zip(source_rows, diagnostics, strict=True)
    ):
        if not isinstance(diagnostic, dict):
            raise ValueError(
                f"native workload diagnostic {source_index} is not an object"
            )
        expected = {
            "workload_index": source_index,
            "source_line": source_line,
            "uuid": source_row.get("uuid"),
            "axes": source_row.get("axes"),
            "workload_sha256": _sha256(source_bytes),
            "workload_sha256_scope": ROW_SHA256_SCOPE,
        }
        for field, value in expected.items():
            if diagnostic.get(field) != value:
                raise ValueError(
                    f"native workload diagnostic identity mismatch at "
                    f"row {source_index} field {field}"
                )
        statuses.append(_classify_bound_native_diagnostic(diagnostic))
    if sr.HOST_CAPACITY_OOM not in statuses:
        raise ValueError("host-capacity cell has no row-level OOM diagnostic")
    return statuses


def plan_proxy_cohort(cell: dict[str, Any], benchmark_dir: Path) -> dict[str, Any]:
    """Pure, read-only geometry planner for one host-capacity problem."""
    definition_path = benchmark_dir / "definition.json"
    workload_path = benchmark_dir / "workload.jsonl"
    definition_bytes = definition_path.read_bytes()
    definition = json.loads(definition_bytes)
    if not isinstance(definition, dict):
        raise ValueError("definition root must be an object")
    expected_name = cell["slug"].removeprefix("problem_")
    if definition.get("name") != expected_name:
        raise ValueError(
            f"definition/problem identity mismatch: {definition.get('name')!r} "
            f"!= {expected_name!r}")
    source_rows = _load_source_rows(workload_path)
    native_run_statuses = _bind_native_diagnostics(cell, source_rows)
    definition_ref = _relative_or_absolute(definition_path)
    definition_sha = _sha256(definition_bytes)
    spec = MODEL_SPECS.get(cell["slug"])
    if spec is None:
        reduced_definition_bytes, definition_delta = None, []
    else:
        reduced_definition_bytes, definition_delta = _definition_variant(
            definition, spec
        )
    reduced_definition_sha = (
        _sha256(reduced_definition_bytes)
        if reduced_definition_bytes is not None else None
    )
    reduced_definition_ref = (
        _external_definition_name(cell, reduced_definition_sha)
        if reduced_definition_sha is not None else None
    )
    row_plans: list[dict[str, Any]] = []
    proxy_rows: dict[str, bytes] = {}
    for index, ((row, raw, _source_line), native_run_status) in enumerate(
        zip(source_rows, native_run_statuses, strict=True)
    ):
        plan, reduced = _plan_row(
            cell=cell, source_index=index, source_row=row, source_bytes=raw,
            native_run_status=native_run_status,
            definition=definition, definition_ref=definition_ref,
            definition_sha256=definition_sha,
            reduced_definition_ref=reduced_definition_ref,
            reduced_definition_sha256=reduced_definition_sha,
            definition_delta=definition_delta)
        row_plans.append(plan)
        if reduced is not None:
            proxy_rows[plan["variant_id"]] = reduced
    cohort_identity = {
        "claim_scope": CLAIM_SCOPE,
        "native_definition_sha256": definition_sha,
        "native_workload_sha256": _sha256(workload_path.read_bytes()),
        "reduced_definition_sha256": reduced_definition_sha,
        "row_plan_variant_ids": [row["variant_id"] for row in row_plans],
        "slug": cell["slug"],
    }
    return {
        "cohort_id": _sha256(canonical_json_bytes(cohort_identity)),
        "definition_ref": definition_ref,
        "definition_sha256": definition_sha,
        "definition_delta": definition_delta,
        "native_workload_ref": _relative_or_absolute(workload_path),
        "native_workload_sha256": _sha256(workload_path.read_bytes()),
        "reduced_definition_ref": reduced_definition_ref,
        "reduced_definition_sha256": reduced_definition_sha,
        "row_plans": row_plans,
        "source_row_count": len(source_rows),
        "_reduced_definition": reduced_definition_bytes,
        "_proxy_rows": proxy_rows,
    }


def classify_shrink_result(*, reduced_ran: bool, status: str,
                           has_safe_axis: bool) -> tuple[str, str]:
    if not has_safe_axis:
        return CLASS_UNAVAILABLE, "no modeled safe geometry axis"
    if not reduced_ran:
        return CLASS_CALIBRATION_INCONCLUSIVE, "proxy correctness run did not complete"
    if status == sr.RUNNABLE:
        return CLASS_REDUCED, "reviewed reduced proxy passes correctness"
    if status == sr.HOST_CAPACITY_OOM:
        return CLASS_UNAVAILABLE, "reviewed reduced proxy remains CUDA-OOM"
    if status == sr.CORRECTNESS_FAILED:
        return CLASS_CORRECTNESS_FAILED, "reviewed reduced proxy fails correctness"
    return CLASS_CALIBRATION_INCONCLUSIVE, f"proxy run {status}"


def host_capacity_slugs(runnability: dict | None) -> list[dict]:
    if not runnability:
        return []
    return [cell for cell in runnability.get("cells", [])
            if cell.get("status") == sr.HOST_CAPACITY_OOM]


def _run_reduced(
    cell: dict[str, Any], reduced_workload: Path, timeout: float,
    probe_contract: dict[str, Any],
    reduced_definition: Path | None = None,
) -> dict[str, Any]:
    """Use the exact same certified module/config/toolchain API as native."""
    return sr.run_certified(
        cell, timeout=timeout, probe_contract=probe_contract,
        workload_path=reduced_workload,
        definition_path=reduced_definition,
    )


def _portable_external_execution_evidence(
    evidence: dict[str, Any], root_identity: Mapping[str, Any], name: str,
    definition_name: str | None = None,
) -> dict[str, Any]:
    """Replace machine-local external paths with one owner-relative filename."""
    expected_path = _external_output_path(root_identity, name).as_posix()
    identity = evidence.get("workload_input")
    state = evidence.get("execution_input_state")
    command = evidence.get("command")
    if (
        not isinstance(identity, dict)
        or identity.get("path") != expected_path
        or not isinstance(state, dict)
        or not isinstance(state.get("workload"), dict)
        or state["workload"].get("path") != expected_path
        or not isinstance(command, list)
        or len(command) != 14
        or command[6] != expected_path
        or _sha256(_read_bound_direct_child(root_identity, name))
        != identity.get("sha256")
    ):
        raise sr.PrerequisiteContractError(
            f"proxy execution evidence is not bound to external child {name}"
        )
    portable = json.loads(json.dumps(evidence))
    portable["workload_input"]["path"] = name
    portable["execution_input_state"]["workload"]["path"] = name
    portable["execution_input_state_sha256"] = sr._canonical_sha256(
        portable["execution_input_state"]
    )
    portable["command"][6] = name
    if definition_name is not None:
        expected_definition_path = _external_output_path(
            root_identity, definition_name
        ).as_posix()
        definition_identity = evidence.get("definition_input")
        definition_state = state.get("definition")
        if (
            not isinstance(definition_identity, dict)
            or definition_identity.get("path") != expected_definition_path
            or not isinstance(definition_state, dict)
            or definition_state.get("path") != expected_definition_path
            or command[4] != expected_definition_path
            or _sha256(_read_bound_direct_child(root_identity, definition_name))
            != definition_identity.get("sha256")
        ):
            raise sr.PrerequisiteContractError(
                "proxy execution evidence is not bound to external definition "
                f"child {definition_name}"
            )
        portable["definition_input"]["path"] = definition_name
        portable["execution_input_state"]["definition"]["path"] = definition_name
        portable["execution_input_state_sha256"] = sr._canonical_sha256(
            portable["execution_input_state"]
        )
        portable["command"][4] = definition_name
    return portable


def _safe_proxy_path(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    resolved_root = root.resolve()
    if resolved_root not in path.parents:
        raise ValueError(f"proxy path escapes output root: {path}")
    return path


def _external_variant_name(cell: Mapping[str, Any], variant_id: str) -> str:
    return f"{cell['category']}__{cell['slug']}__variant__{variant_id}.jsonl"


def _external_definition_name(
    cell: Mapping[str, Any], definition_sha256: str,
) -> str:
    return (
        f"{cell['category']}__{cell['slug']}__definition__"
        f"{definition_sha256}.json"
    )


def _external_cohort_name(cell: Mapping[str, Any], cohort_id: str) -> str:
    return f"{cell['category']}__{cell['slug']}__cohort__{cohort_id}.jsonl"


def _public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if not key.startswith("_")}


def _assert_execution_evidence(
    evidence: Any,
    expected_rows: list[tuple[str, str]],
) -> None:
    """Require a complete UUID -> exact reduced-row-byte PASS bijection."""
    if len(expected_rows) != len({uuid for uuid, _ in expected_rows}):
        raise ValueError("expected proxy evidence repeats a workload UUID")
    diagnostics = evidence.get("workload_diagnostics") if isinstance(evidence, dict) else None
    if (
        not isinstance(evidence, dict)
        or evidence.get("status") != sr.RUNNABLE
        or evidence.get("execution_kind") != "proxy"
        or evidence.get("diagnostics_complete") is not True
        or evidence.get("expected_workloads") != len(expected_rows)
        or evidence.get("returned_workloads") != len(expected_rows)
        or evidence.get("passed_workloads") != len(expected_rows)
        or evidence.get("unexpected_workload_diagnostics") != []
        or not isinstance(diagnostics, list)
        or len(diagnostics) != len(expected_rows)
    ):
        raise ValueError("proxy execution evidence is incomplete")
    expected = dict(expected_rows)
    observed: dict[str, str] = {}
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            raise ValueError("proxy execution diagnostic is not an object")
        uuid = diagnostic.get("uuid")
        digest = diagnostic.get("workload_sha256")
        if not isinstance(uuid, str) or uuid in observed:
            raise ValueError("proxy execution evidence repeats a UUID")
        if (
            diagnostic.get("diagnostic_present") is not True
            or str(diagnostic.get("status", "")).lower() != "passed"
            or diagnostic.get("workload_sha256_scope") != ROW_SHA256_SCOPE
            or not isinstance(digest, str)
        ):
            raise ValueError("proxy execution diagnostic is not a bound PASS")
        observed[uuid] = digest
    if observed != expected:
        raise ValueError("proxy execution UUID/hash evidence is not bijective")


def _assert_proxy_row_evidence(
        row: dict[str, Any], payload: bytes, *, require_passed: bool) -> None:
    """Fail closed before a proxy row can enter a reduced cohort.

    Unchanged or unverified rows may remain diagnostic planning evidence, but
    they can never be selected or labeled as reduced evidence.
    """
    if row.get("planning_status") != "proxy_planned":
        raise ValueError("non-proxy row selected for reduced cohort")
    if (row.get("claim_scope") != CLAIM_SCOPE
            or row.get("native_predictive") is not False):
        raise ValueError("proxy row carries an invalid claim scope")
    if (not row.get("slug") or not row.get("source_workload_uuid")
            or not row.get("variant_id")):
        raise ValueError("proxy row lacks its provenance identity tuple")
    axes_delta = row.get("axes_delta")
    definition_delta = row.get("definition_delta")
    if not isinstance(axes_delta, list) or not isinstance(definition_delta, list):
        raise ValueError("proxy row lacks closed axes/definition deltas")
    if not axes_delta and not definition_delta:
        raise ValueError("unchanged proxy bundle selected for reduced cohort")
    if not axes_delta and row.get("native_axes") != row.get("reduced_axes"):
        raise ValueError("definition-only proxy silently changed workload axes")
    if axes_delta and row.get("native_axes") == row.get("reduced_axes"):
        raise ValueError("workload proxy delta leaves workload axes unchanged")
    native_sha = row.get("native_row_sha256")
    reduced_sha = row.get("reduced_row_sha256")
    if not isinstance(native_sha, str) or not isinstance(reduced_sha, str):
        raise ValueError("proxy row lacks native/reduced byte hashes")
    native_definition_sha = row.get("definition_sha256")
    reduced_definition_sha = row.get("reduced_definition_sha256")
    reduced_definition_ref = row.get("reduced_definition_ref")
    if definition_delta:
        if (
            not isinstance(native_definition_sha, str)
            or not isinstance(reduced_definition_sha, str)
            or not isinstance(reduced_definition_ref, str)
            or native_definition_sha == reduced_definition_sha
        ):
            raise ValueError("definition proxy lacks distinct bound definition bytes")
    elif reduced_definition_sha is not None or reduced_definition_ref is not None:
        raise ValueError("workload-only proxy carries an undeclared definition variant")
    if native_sha == reduced_sha and not definition_delta:
        raise ValueError("proxy bundle bytes are unchanged from native")
    model = row.get("model")
    model_id = model.get("model_id") if isinstance(model, dict) else None
    if (
        not isinstance(model_id, str)
        or row.get("variant_id") != _variant_id(
            slug=row["slug"], source_uuid=row["source_workload_uuid"],
            native_sha256=native_sha, outcome="proxy_planned",
            reduced_sha256=reduced_sha, model_id=model_id,
            native_definition_sha256=native_definition_sha,
            reduced_definition_sha256=reduced_definition_sha,
        )
    ):
        raise ValueError("proxy variant identity does not bind the exact bundle")
    if _sha256(payload) != reduced_sha:
        raise ValueError("proxy row payload differs from planned reduced bytes")
    if require_passed and (
            row.get("correctness_result") != "passed"
            or row.get("reduced_run_status") != sr.RUNNABLE):
        raise ValueError("unverified proxy row selected for reduced cohort")
    if require_passed:
        evidence = row.get("execution_evidence")
        _assert_execution_evidence(
            evidence, [(row["source_workload_uuid"], reduced_sha)]
        )


def _assert_reduced_cohort_evidence(
        plan: dict[str, Any], selected: list[dict[str, Any]]) -> None:
    if not selected:
        raise ValueError("reduced cohort is empty")
    identities = [
        (row.get("slug"), row.get("source_workload_uuid"), row.get("variant_id"))
        for row in selected
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("reduced cohort repeats a provenance identity tuple")
    for row in selected:
        payload = plan.get("_proxy_rows", {}).get(row["variant_id"])
        if not isinstance(payload, bytes):
            raise ValueError("reduced cohort payload is absent")
        _assert_proxy_row_evidence(row, payload, require_passed=True)


def _record(cell: dict[str, Any], plan: dict[str, Any], *, classification: str,
            detail: str, reduced_ref: str | None, reduced_sha256: str | None,
            reduced_status: str | None, reduced_detail: str | None,
            correctness: str,
            cohort_execution_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    row_plans = [_public_plan(row) for row in plan.get("row_plans", [])]
    return {
        "claim_scope": CLAIM_SCOPE,
        "classification": classification,
        "cohort_id": plan.get("cohort_id"),
        "cohort_execution_evidence": cohort_execution_evidence,
        "correctness_result": correctness,
        "category": cell.get("category"),
        "definition_ref": plan.get("definition_ref"),
        "definition_sha256": plan.get("definition_sha256"),
        "definition_delta": plan.get("definition_delta", []),
        "detail": detail,
        "halvings": None,
        "l2_side_verdict": "per_variant_geometry",
        "language": cell.get("language"),
        "native_predictive": False,
        "native_shape_ref": plan.get("native_workload_ref"),
        "native_workload_sha256": plan.get("native_workload_sha256"),
        "occupancy_wave_verdict": "per_variant_geometry",
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "reduced_run_detail": reduced_detail,
        "reduced_run_status": reduced_status,
        "reduced_definition_ref": plan.get("reduced_definition_ref"),
        "reduced_definition_sha256": plan.get("reduced_definition_sha256"),
        "reduced_workload_ref": reduced_ref,
        "reduced_workload_sha256": reduced_sha256,
        "replay_mode_check": REPLAY_MODE_CHECK,
        "roofline_class_verdict": "per_variant_geometry",
        "row_plans": row_plans,
        "shrink_vector": [
            {"source_workload_uuid": row["source_workload_uuid"],
             "variant_id": row["variant_id"], "axes_delta": row["axes_delta"]}
            for row in row_plans if row["planning_status"] == "proxy_planned"
        ],
        "slug": cell["slug"],
        "source_row_count": plan.get("source_row_count", 0),
    }


def _empty_plan(cell: dict[str, Any], detail: str) -> dict[str, Any]:
    return {
        "cohort_id": _sha256(canonical_json_bytes(
            {"slug": cell["slug"], "error": detail})),
        "definition_ref": None, "definition_sha256": None,
        "definition_delta": [],
        "reduced_definition_ref": None,
        "reduced_definition_sha256": None,
        "native_workload_ref": None, "native_workload_sha256": None,
        "row_plans": [], "source_row_count": 0,
        "_reduced_definition": None, "_proxy_rows": {},
    }


def apply_one(
    cell: dict[str, Any], timeout: float,
    probe_contract: dict[str, Any],
    *,
    reduced_output_root_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan then correctness-check one problem without mutating native inputs."""
    try:
        _, _, benchmark_dir = sr.resolve_paths(cell)
        plan = plan_proxy_cohort(cell, benchmark_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        detail = f"proxy planning failed: {exc}"
        return _record(
            cell, _empty_plan(cell, detail),
            classification=CLASS_CALIBRATION_INCONCLUSIVE, detail=detail,
            reduced_ref=None, reduced_sha256=None,
            reduced_status=sr.RESOLVE_ERROR, reduced_detail=detail,
            correctness="not_reached")

    statuses = Counter(row["planning_status"] for row in plan["row_plans"])
    if statuses["needs_model"]:
        detail = f"{statuses['needs_model']} source workload row(s) need a family model"
        return _record(
            cell, plan, classification=CLASS_NEEDS_MODEL, detail=detail,
            reduced_ref=None, reduced_sha256=None,
            reduced_status="needs_model", reduced_detail=detail,
            correctness="not_reached")
    if statuses["geometry_unavailable"]:
        detail = (f"{statuses['geometry_unavailable']} source workload row(s) have "
                  "no safe fitting modeled proxy")
        return _record(
            cell, plan, classification=CLASS_UNAVAILABLE, detail=detail,
            reduced_ref=None, reduced_sha256=None,
            reduced_status="geometry_unavailable", reduced_detail=detail,
            correctness="not_reached")

    planned = [row for row in plan["row_plans"]
               if row["planning_status"] == "proxy_planned"]
    if not planned:
        detail = "host-capacity status has no modeled row requiring a proxy"
        return _record(
            cell, plan, classification=CLASS_NEEDS_MODEL, detail=detail,
            reduced_ref=None, reduced_sha256=None,
            reduced_status="needs_model", reduced_detail=detail,
            correctness="not_reached")

    reduced_definition_path: Path | None = None
    reduced_definition_name: str | None = None
    reduced_definition_payload = plan.get("_reduced_definition")
    if reduced_definition_payload is not None:
        if not isinstance(reduced_definition_payload, bytes):
            raise ValueError("planned reduced definition payload is not bytes")
        reduced_definition_name = plan.get("reduced_definition_ref")
        if (
            not isinstance(reduced_definition_name, str)
            or _sha256(reduced_definition_payload)
            != plan.get("reduced_definition_sha256")
            or reduced_definition_name != _external_definition_name(
                cell, plan["reduced_definition_sha256"]
            )
        ):
            raise ValueError("planned reduced definition identity is inconsistent")
        if reduced_output_root_identity is None:
            reduced_definition_path = _safe_proxy_path(
                PROXY_VARIANT_ROOT, str(cell["category"]), cell["slug"],
                reduced_definition_name,
            )
            sr.atomic_write_bytes(reduced_definition_path, reduced_definition_payload)
        else:
            reduced_definition_path = _publish_immutable_or_verify(
                reduced_output_root_identity, reduced_definition_name,
                reduced_definition_payload,
            )

    results: list[dict[str, Any]] = []
    for row in planned:
        payload = plan["_proxy_rows"][row["variant_id"]] + b"\n"
        _assert_proxy_row_evidence(row, payload[:-1], require_passed=False)
        if reduced_output_root_identity is None:
            variant_path = _safe_proxy_path(
                PROXY_VARIANT_ROOT, str(cell["category"]), cell["slug"],
                f"{row['variant_id']}.jsonl")
            sr.atomic_write_bytes(variant_path, payload)
        else:
            variant_path = _publish_immutable_or_verify(
                reduced_output_root_identity,
                _external_variant_name(cell, row["variant_id"]), payload,
            )
        result = _run_reduced(
            cell, variant_path, timeout, probe_contract,
            reduced_definition_path,
        )
        if reduced_output_root_identity is not None:
            result = _portable_external_execution_evidence(
                result, reduced_output_root_identity,
                _external_variant_name(cell, row["variant_id"]),
                reduced_definition_name,
            )
        row["reduced_run_status"] = result["status"]
        row["reduced_run_detail"] = result["detail"]
        row["execution_evidence"] = result
        if result["status"] == sr.RUNNABLE:
            row["correctness_result"] = "passed"
        elif result["status"] == sr.CORRECTNESS_FAILED:
            row["correctness_result"] = "failed"
        else:
            row["correctness_result"] = "not_reached"
        results.append(result)

    if any(result["status"] != sr.RUNNABLE for result in results):
        if any(result["status"] == sr.CORRECTNESS_FAILED for result in results):
            classification = CLASS_CORRECTNESS_FAILED
            correctness = "failed"
        elif any(result["status"] == sr.HOST_CAPACITY_OOM for result in results):
            classification = CLASS_UNAVAILABLE
            correctness = "not_reached"
        else:
            classification = CLASS_CALIBRATION_INCONCLUSIVE
            correctness = "not_reached"
        detail = "one or more per-variant proxy correctness runs failed"
        return _record(
            cell, plan, classification=classification, detail=detail,
            reduced_ref=None, reduced_sha256=None,
            reduced_status="variant_failure", reduced_detail=detail,
            correctness=correctness)

    planned.sort(key=lambda row: (row["source_workload_index"], row["variant_id"]))
    _assert_reduced_cohort_evidence(plan, planned)
    cohort_payload = b"".join(
        plan["_proxy_rows"][row["variant_id"]] + b"\n" for row in planned)
    if reduced_output_root_identity is None:
        cohort_path = _safe_proxy_path(
            REDUCED_WORKLOAD_ROOT, str(cell["category"]),
            f"{cell['slug']}__geometry_proxy_v1.jsonl")
        sr.atomic_write_bytes(cohort_path, cohort_payload)
    else:
        cohort_path = _publish_immutable_or_verify(
            reduced_output_root_identity,
            _external_cohort_name(cell, str(plan["cohort_id"])), cohort_payload,
        )
    cohort_result = _run_reduced(
        cell, cohort_path, timeout, probe_contract,
        reduced_definition_path,
    )
    if reduced_output_root_identity is not None:
        cohort_result = _portable_external_execution_evidence(
            cohort_result, reduced_output_root_identity,
            _external_cohort_name(cell, str(plan["cohort_id"])),
            reduced_definition_name,
        )
    if cohort_result["status"] != sr.RUNNABLE:
        if cohort_result["status"] == sr.CORRECTNESS_FAILED:
            classification = CLASS_CORRECTNESS_FAILED
            correctness = "failed"
        elif cohort_result["status"] == sr.HOST_CAPACITY_OOM:
            classification = CLASS_UNAVAILABLE
            correctness = "not_reached"
        else:
            classification = CLASS_CALIBRATION_INCONCLUSIVE
            correctness = "not_reached"
        detail = "combined geometry-proxy cohort correctness run failed"
        return _record(
            cell, plan, classification=classification, detail=detail,
            reduced_ref=None, reduced_sha256=None,
            reduced_status=cohort_result["status"],
            reduced_detail=cohort_result["detail"], correctness=correctness,
            cohort_execution_evidence=cohort_result)
    _assert_execution_evidence(
        cohort_result,
        [
            (row["source_workload_uuid"], row["reduced_row_sha256"])
            for row in planned
        ],
    )
    detail = (f"{len(planned)}/{len(planned)} reviewed reduced proxy variants "
              "and the combined proxy cohort passed correctness")
    return _record(
        cell, plan, classification=CLASS_REDUCED, detail=detail,
        reduced_ref=(
            cohort_path.name if reduced_output_root_identity is not None
            else _relative_or_absolute(cohort_path)
        ),
        reduced_sha256=_sha256(cohort_payload), reduced_status=sr.RUNNABLE,
        reduced_detail=detail, correctness="passed",
        cohort_execution_evidence=cohort_result)


def _source_closure(targets: list[dict[str, Any]]) -> list[dict[str, str]]:
    paths: set[Path] = set()
    excluded_parts = {"__pycache__", "build", ".cache", ".venv"}
    if DEFERRED_JSON.is_symlink() or not DEFERRED_JSON.is_file():
        raise sr.PrerequisiteContractError(
            f"SOL shrink deferred-ledger input is nonregular: {DEFERRED_JSON}"
        )
    paths.add(DEFERRED_JSON.resolve())
    for cell in targets:
        baseline, _, benchmark = sr.resolve_paths(cell)
        for root in (baseline, benchmark):
            for path in root.rglob("*"):
                if any(part in excluded_parts for part in path.relative_to(root).parts):
                    continue
                if path.is_symlink():
                    raise sr.PrerequisiteContractError(
                        f"SOL shrink source closure contains symlink: {path}"
                    )
                if path.is_file():
                    paths.add(path.resolve())
    rows = []
    for path in sorted(paths, key=lambda value: value.relative_to(REPO_ROOT).as_posix()):
        rows.append({
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sr._sha256_file(path),
        })
    return rows


def _bound_root_names(root_identity: Mapping[str, Any]) -> set[str]:
    descriptor = _open_bound_root(root_identity)
    try:
        return set(os.listdir(descriptor))
    finally:
        os.close(descriptor)


def _owner_marker_document(
    plan: Mapping[str, Any], root_identity: Mapping[str, Any],
) -> dict[str, Any]:
    document = {
        "schema_version": 1,
        "purpose": EXTERNAL_OWNER_PURPOSE,
        "source_commit": plan["source_commit"],
        "source_worktree_state": plan["source_worktree_state"],
        "source_worktree_state_sha256": plan[
            "source_worktree_state_sha256"
        ],
        "runnability_name": plan["runnability_name"],
        "runnability_sha256": plan["runnability_sha256"],
        "runnability_resume_identity_sha256": plan[
            "runnability_resume_identity_sha256"
        ],
        "runnability_external_campaign_owner": plan[
            "runnability_external_campaign_owner"
        ],
        "probe_config_path": sr._path_reference(sr.PROBE_CONFIG),
        "probe_config_sha256": sr.PROBE_CONFIG_SHA256,
        "probe_contract_sha256": plan["probe_contract_sha256"],
        "native_env_hash": plan["native_env_hash"],
        "target_set_sha256": plan["target_set_sha256"],
        "source_closure_sha256": plan["source_closure_sha256"],
        "shrink_plan_identity_sha256": plan["shrink_plan_identity_sha256"],
        "provenance_output_name": plan["provenance_output_name"],
        "checkpoint_output_name": plan["checkpoint_output_name"],
    }
    document["owner_seal_sha256"] = _sha256(canonical_json_bytes(document))
    return document


def _bind_external_owner(
    plan: dict[str, Any], root_identity: Mapping[str, Any], *, mode: str,
) -> dict[str, Any]:
    if mode not in {"ensure", "verify"}:
        raise ValueError(f"unknown owner marker mode: {mode}")
    document = _owner_marker_document(plan, root_identity)
    payload = canonical_json_bytes(document)
    marker_path = _external_output_path(
        root_identity, EXTERNAL_OWNER_MARKER_NAME
    )
    if mode == "ensure":
        names = _bound_root_names(root_identity)
        if EXTERNAL_OWNER_MARKER_NAME not in names and names:
            raise sr.PrerequisiteContractError(
                "fresh SOL shrink root is nonempty and has no owner marker: "
                f"{sorted(names)}"
            )
        _publish_immutable_or_verify(
            root_identity, EXTERNAL_OWNER_MARKER_NAME, payload
        )
    else:
        observed = _read_bound_direct_child(
            root_identity, EXTERNAL_OWNER_MARKER_NAME
        )
        if observed != payload:
            raise sr.PrerequisiteContractError(
                "external reduced-output owner marker identity mismatch"
            )
    return {
        "name": marker_path.name,
        "sha256": _sha256(payload),
        "owner_seal_sha256": document["owner_seal_sha256"],
    }


def build_shrink_contract(
    *,
    runnability_sha256: str,
    runnability: dict[str, Any],
    probe_contract: dict[str, Any],
    targets: list[dict[str, Any]],
    runnability_path: Path,
    reduced_output_root: Path,
    provenance_output_path: Path,
    owner_mode: str = "ensure",
) -> dict[str, Any]:
    target_rows = sorted(
        [
            {"category": cell["category"], "slug": cell["slug"],
             "language": cell["language"]}
            for cell in targets
        ],
        key=lambda row: (row["category"], row["slug"], row["language"]),
    )
    if len({(row["category"], row["slug"], row["language"])
            for row in target_rows}) != len(target_rows):
        raise sr.PrerequisiteContractError("shrink target set contains duplicates")
    source_closure = _source_closure(targets)
    target_set_sha = _sha256(canonical_json_bytes({"targets": target_rows}))
    source_closure_sha = _sha256(
        canonical_json_bytes({"files": source_closure}))
    public_probe = sr.public_probe_contract(probe_contract)
    worktree_state = sr.source_worktree_state()
    root_identity = validate_external_reduced_output_root(reduced_output_root)
    output_path = Path(provenance_output_path)
    if (
        not output_path.is_absolute()
        or output_path.absolute()
        != _external_output_path(root_identity, output_path.name)
    ):
        raise sr.PrerequisiteContractError(
            "shrink provenance --out must be a direct child of reduced output root"
        )
    checkpoint_path = _external_output_path(
        root_identity, f"{output_path.name}.partial"
    )
    requested_runnability = Path(runnability_path)
    if not requested_runnability.is_absolute():
        raise sr.PrerequisiteContractError(
            "shrink runnability input must be an explicit absolute path"
        )
    runnability_root_identity = sr.validate_external_campaign_root(
        requested_runnability.parent
    )
    resolved_runnability = sr.external_child_path(
        runnability_root_identity, requested_runnability.name
    )
    runnability_payload = sr.read_external_child(
        runnability_root_identity, resolved_runnability.name
    )
    if (
        requested_runnability.absolute() != resolved_runnability
        or runnability.get("external_output_name") != resolved_runnability.name
        or _sha256(runnability_payload) != runnability_sha256
    ):
        raise sr.PrerequisiteContractError(
            "shrink runnability input is not its sealed owner-root direct child"
        )
    current_commit = _source_head()
    if runnability.get("source_commit") != current_commit:
        raise sr.PrerequisiteContractError(
            "shrink source commit differs from external runnability source commit"
        )
    public = {
        "shrink_contract_version": 1,
        "source_commit": current_commit,
        "source_worktree_state": worktree_state,
        "source_worktree_state_sha256": _sha256(
            canonical_json_bytes(worktree_state)
        ),
        "native_env_hash": runnability["native_env_hash"],
        "runnability_name": resolved_runnability.name,
        "runnability_sha256": runnability_sha256,
        "runnability_resume_identity_sha256":
            runnability["resume_identity_sha256"],
        "runnability_external_campaign_owner": runnability[
            "external_campaign_owner"
        ],
        "probe_contract_sha256": _sha256(canonical_json_bytes(public_probe)),
        "producer_sha256": sr._sha256_file(Path(__file__)),
        "calculator_sha256": sr._sha256_file(Path(sc.__file__)),
        "provenance_schema_sha256": sr._sha256_file(PROVENANCE_SCHEMA),
        "artifact_schema_sha256": sr._sha256_file(ARTIFACT_SCHEMA),
        "model_registry_sha256": _sha256(canonical_json_bytes(MODEL_SPECS)),
        "model_threshold_sha256": _sha256(canonical_json_bytes({
            "hardware": sc.HW,
            "thresholds": sc.TH,
            "classes": [
                sc.CLASS_SHAPE_ADJUSTED, sc.CLASS_REDUCED,
                sc.CLASS_UNAVAILABLE, sc.CLASS_NATIVE_REPLAY,
            ],
        })),
        "target_count": len(target_rows),
        "target_set_sha256": target_set_sha,
        "targets": target_rows,
        "source_closure_file_count": len(source_closure),
        "source_closure_sha256": source_closure_sha,
        "source_closure": source_closure,
        "provenance_output_name": output_path.name,
        "checkpoint_output_name": checkpoint_path.name,
    }
    public["shrink_plan_identity_sha256"] = _sha256(
        canonical_json_bytes(public)
    )
    public["reduced_output_owner"] = _bind_external_owner(
        public, root_identity, mode=owner_mode
    )
    public["shrink_resume_identity_sha256"] = _sha256(
        canonical_json_bytes(public))
    public["_reduced_output_root_identity"] = root_identity
    return public


def _validate_v3_cell_schema(record: dict[str, Any]) -> None:
    schema = json.loads(ARTIFACT_SCHEMA.read_text())
    jsonschema.validate(
        record,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/v3Cell",
            "$defs": schema["$defs"],
        },
    )


def _execution_input_identities_match(
    evidence: dict[str, Any], *,
    reduced_output_root_identity: Mapping[str, Any] | None = None,
    expected_workload_name: str | None = None,
    expected_definition_name: str | None = None,
) -> bool:
    input_state = evidence.get("execution_input_state")
    if (
        not isinstance(input_state, dict)
        or set(input_state) != {"definition", "workload", "solution"}
        or evidence.get("execution_input_state_sha256")
        != sr._canonical_sha256(input_state)
    ):
        return False
    state_names = {
        "definition_input": "definition",
        "workload_input": "workload",
        "solution_input": "solution",
    }
    for field in ("definition_input", "workload_input", "solution_input"):
        identity = evidence.get(field)
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            return False
        external_name = (
            expected_workload_name if field == "workload_input" else
            expected_definition_name if field == "definition_input" else None
        )
        if external_name is not None and reduced_output_root_identity is not None:
            if not _bound_external_file_matches(
                reduced_output_root_identity, identity.get("path"), external_name,
                identity.get("sha256"),
            ):
                return False
        else:
            path = Path(identity["path"])
            if not path.is_absolute():
                path = REPO_ROOT / path
            try:
                if sr._path_identity(path) != identity:
                    return False
            except sr.PrerequisiteContractError:
                return False
        state = input_state.get(state_names[field])
        if not isinstance(state, dict) or state != {
            "path": identity["path"], "state": "present",
            "sha256": identity["sha256"], "detail": None,
        }:
            return False
    return True


def _bound_external_file_matches(
    root_identity: Mapping[str, Any], reference: Any, expected_name: str,
    expected_sha256: Any,
) -> bool:
    try:
        _external_output_path(root_identity, expected_name)
        if reference != expected_name:
            return False
        payload = _read_bound_direct_child(root_identity, expected_name)
        return isinstance(expected_sha256, str) and _sha256(payload) == expected_sha256
    except (OSError, KeyError, TypeError, sr.PrerequisiteContractError):
        return False


def _record_is_resumable(
    record: dict[str, Any], expected_slug: str, *,
    reduced_output_root_identity: Mapping[str, Any] | None = None,
) -> bool:
    try:
        row_plans = record.get("row_plans")
        source_row_count = record.get("source_row_count")
        if (
            record.get("slug") != expected_slug
            or record.get("provenance_contract_version")
            != PROVENANCE_CONTRACT_VERSION
            or record.get("claim_scope") != CLAIM_SCOPE
            or record.get("native_predictive") is not False
            or not isinstance(row_plans, list)
            or not isinstance(source_row_count, int)
            or isinstance(source_row_count, bool)
            or source_row_count != len(row_plans)
            or any(not isinstance(row, dict) for row in row_plans)
            or sorted(row.get("source_workload_index") for row in row_plans)
            != list(range(source_row_count))
            or len({row.get("source_workload_uuid") for row in row_plans})
            != source_row_count
        ):
            return False
        _validate_v3_cell_schema(record)
        planned = [
            row for row in row_plans
            if row.get("planning_status") == "proxy_planned"
        ]
        for row in planned:
            if reduced_output_root_identity is None:
                variant_path = _safe_proxy_path(
                    PROXY_VARIANT_ROOT, str(record.get("category")), expected_slug,
                    f"{row['variant_id']}.jsonl",
                )
                variant_payload = variant_path.read_bytes()
            else:
                variant_name = _external_variant_name(record, row["variant_id"])
                evidence = row.get("execution_evidence")
                workload_identity = (
                    evidence.get("workload_input")
                    if isinstance(evidence, dict) else None
                )
                if (
                    not isinstance(workload_identity, dict)
                    or not _bound_external_file_matches(
                        reduced_output_root_identity,
                        workload_identity.get("path"), variant_name,
                        workload_identity.get("sha256"),
                    )
                ):
                    return False
                variant_payload = _read_bound_direct_child(
                    reduced_output_root_identity, variant_name
                )
            if (
                not variant_payload.endswith(b"\n")
                or b"\n" in variant_payload[:-1]
                or _sha256(variant_payload[:-1]) != row.get("reduced_row_sha256")
                or not isinstance(row.get("execution_evidence"), dict)
                or row["execution_evidence"].get("status")
                != row.get("reduced_run_status")
                or not _execution_input_identities_match(
                    row["execution_evidence"],
                    reduced_output_root_identity=reduced_output_root_identity,
                    expected_workload_name=(
                        _external_variant_name(record, row["variant_id"])
                        if reduced_output_root_identity is not None else None
                    ),
                    expected_definition_name=(
                        row.get("reduced_definition_ref")
                        if reduced_output_root_identity is not None else None
                    ),
                )
            ):
                return False
            _assert_proxy_row_evidence(
                row, variant_payload[:-1],
                require_passed=row.get("correctness_result") == "passed",
            )
        if record.get("classification") != CLASS_REDUCED:
            cohort_evidence = record.get("cohort_execution_evidence")
            if cohort_evidence is not None and (
                not isinstance(cohort_evidence, dict)
                or cohort_evidence.get("status") != record.get("reduced_run_status")
                or not _execution_input_identities_match(
                    cohort_evidence,
                    reduced_output_root_identity=reduced_output_root_identity,
                    expected_workload_name=(
                        _external_cohort_name(
                            record, str(record.get("cohort_id"))
                        ) if reduced_output_root_identity is not None else None
                    ),
                    expected_definition_name=(
                        record.get("reduced_definition_ref")
                        if reduced_output_root_identity is not None else None
                    ),
                )
            ):
                return False
            return (
                record.get("reduced_workload_ref") is None
                and record.get("reduced_workload_sha256") is None
                and record.get("correctness_result") in {"failed", "not_reached"}
            )
        cohort_ref = record.get("reduced_workload_ref")
        if not isinstance(cohort_ref, str):
            return False
        if reduced_output_root_identity is None:
            cohort_path = Path(cohort_ref)
            if not cohort_path.is_absolute():
                cohort_path = REPO_ROOT / cohort_path
            payload = cohort_path.read_bytes()
            if _sha256(payload) != record.get("reduced_workload_sha256"):
                return False
        else:
            cohort_name = _external_cohort_name(
                record, str(record.get("cohort_id"))
            )
            if not _bound_external_file_matches(
                reduced_output_root_identity, cohort_ref, cohort_name,
                record.get("reduced_workload_sha256"),
            ):
                return False
            payload = _read_bound_direct_child(
                reduced_output_root_identity, cohort_name
            )
        lines = [line for line in payload.split(b"\n") if line.strip()]
        if len(lines) != len(planned) or not planned:
            return False
        by_uuid: dict[str, bytes] = {}
        for line in lines:
            parsed = json.loads(line)
            uuid = parsed.get("uuid") if isinstance(parsed, dict) else None
            if not isinstance(uuid, str) or uuid in by_uuid:
                return False
            by_uuid[uuid] = line
        for row in planned:
            line = by_uuid.get(row.get("source_workload_uuid"))
            if line is None:
                return False
            _assert_proxy_row_evidence(row, line, require_passed=True)
        _assert_execution_evidence(
            record.get("cohort_execution_evidence"),
            [
                (row["source_workload_uuid"], row["reduced_row_sha256"])
                for row in planned
            ],
        )
        if not _execution_input_identities_match(
            record["cohort_execution_evidence"],
            reduced_output_root_identity=reduced_output_root_identity,
            expected_workload_name=(
                _external_cohort_name(record, str(record.get("cohort_id")))
                if reduced_output_root_identity is not None else None
            ),
            expected_definition_name=(
                record.get("reduced_definition_ref")
                if reduced_output_root_identity is not None else None
            ),
        ):
            return False
        return True
    except (
        OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
        jsonschema.ValidationError,
    ):
        return False


def _resumable_checkpoint_records(
    existing: Any, *, shrink_contract: Mapping[str, Any],
    reduced_output_root_identity: Mapping[str, Any] | None = None,
    runnability: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate every checkpoint field/row or refuse resume without salvage."""
    if not isinstance(existing, dict):
        raise sr.PrerequisiteContractError("shrink checkpoint root is not an object")
    required = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "publication_state": "checkpoint",
        **{
            key: value for key, value in shrink_contract.items()
            if not key.startswith("_")
        },
    }
    for field, expected in required.items():
        if existing.get(field) != expected:
            raise sr.PrerequisiteContractError(
                f"shrink checkpoint contract drift at {field}"
            )
    expected_top_keys = set(required) | {
        "external_child_manifest", "cells",
        "prior_host_capacity_reconciliation",
    }
    if set(existing) != expected_top_keys:
        raise sr.PrerequisiteContractError(
            "shrink checkpoint has unknown or missing top-level fields: "
            f"{sorted(set(existing) ^ expected_top_keys)}"
        )
    if not isinstance(existing.get("cells"), list):
        raise sr.PrerequisiteContractError("shrink checkpoint cells are not an array")
    target_slugs = {row["slug"] for row in shrink_contract["targets"]}
    records: dict[str, dict[str, Any]] = {}
    for index, cell in enumerate(existing["cells"]):
        if not isinstance(cell, dict) or not isinstance(cell.get("slug"), str):
            raise sr.PrerequisiteContractError(
                f"shrink checkpoint cell {index} lacks an identity"
            )
        slug = cell["slug"]
        if slug not in target_slugs:
            raise sr.PrerequisiteContractError(
                f"shrink checkpoint contains an out-of-contract target: {slug}"
            )
        if not _record_is_resumable(
            cell, slug,
            reduced_output_root_identity=reduced_output_root_identity,
        ):
            raise sr.PrerequisiteContractError(
                f"shrink checkpoint cell is stale or invalid: {slug}"
            )
        if slug in records:
            raise sr.PrerequisiteContractError(
                f"duplicate shrink checkpoint cell: {slug}"
            )
        records[slug] = cell
    if reduced_output_root_identity is not None and existing.get(
        "external_child_manifest"
    ) != _external_child_manifest(records, reduced_output_root_identity):
        raise sr.PrerequisiteContractError(
            "shrink checkpoint external child manifest does not rederive"
        )
    if runnability is not None and existing.get(
        "prior_host_capacity_reconciliation"
    ) != prior_host_capacity_reconciliation(dict(runnability), records):
        raise sr.PrerequisiteContractError(
            "shrink checkpoint reconciliation does not rederive"
        )
    return records


def _validate_completed_shrink_against_contract(
    artifact: dict[str, Any], *, runnability: dict[str, Any],
    shrink_contract: dict[str, Any],
    reduced_output_root_identity: Mapping[str, Any] | None = None,
) -> None:
    schema = json.loads(ARTIFACT_SCHEMA.read_text())
    jsonschema.validate(artifact, schema)
    required = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": runnability["native_env_hash"],
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "publication_state": "complete",
        **{
            key: value for key, value in shrink_contract.items()
            if not key.startswith("_")
        },
    }
    for field, expected in required.items():
        if artifact.get(field) != expected:
            raise sr.PrerequisiteContractError(
                f"completed shrink provenance drift at {field}"
            )
    cells = artifact.get("cells")
    if not isinstance(cells, list):
        raise sr.PrerequisiteContractError("shrink provenance cells are not an array")
    expected_slugs = {row["slug"] for row in shrink_contract["targets"]}
    records: dict[str, dict[str, Any]] = {}
    for record in cells:
        if not isinstance(record, dict) or not isinstance(record.get("slug"), str):
            raise sr.PrerequisiteContractError("shrink provenance cell identity invalid")
        slug = record["slug"]
        if slug in records:
            raise sr.PrerequisiteContractError(
                f"duplicate shrink provenance cell: {slug}"
            )
        records[slug] = record
        if not _record_is_resumable(
            record, slug,
            reduced_output_root_identity=reduced_output_root_identity,
        ):
            raise sr.PrerequisiteContractError(
                f"shrink provenance evidence is incomplete or stale: {slug}"
            )
    if set(records) != expected_slugs:
        raise sr.PrerequisiteContractError(
            "completed shrink target set mismatch: "
            f"missing={sorted(expected_slugs - set(records))} "
            f"extra={sorted(set(records) - expected_slugs)}"
        )
    if reduced_output_root_identity is not None and artifact.get(
        "external_child_manifest"
    ) != _external_child_manifest(records, reduced_output_root_identity):
        raise sr.PrerequisiteContractError(
            "shrink external child manifest does not rederive"
        )
    expected_reconciliation = prior_host_capacity_reconciliation(
        runnability, records
    )
    if artifact.get("prior_host_capacity_reconciliation") != expected_reconciliation:
        raise sr.PrerequisiteContractError(
            "completed shrink reconciliation does not rederive"
        )


def validate_completed_shrink_provenance(
    artifact: dict[str, Any], *, runnability: dict[str, Any],
    runnability_sha256: str, probe_contract: dict[str, Any],
    targets: list[dict[str, Any]],
    runnability_path: Path | None = None,
    provenance_path: Path | None = None,
    reduced_output_root: Path | None = None,
) -> None:
    """Validate complete geometry-proxy evidence against all current bytes."""
    if runnability_path is None:
        raise sr.PrerequisiteContractError(
            "external runnability path is required for shrink validation"
        )
    selected_runnability_path = Path(runnability_path)
    sr.validate_completed_runnability(
        runnability, expected_contract=probe_contract,
        require_external_owner=True,
        external_campaign_root=selected_runnability_path.parent,
    )
    def target_identity(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
        return sorted(
            (row["category"], row["slug"], row["language"]) for row in rows
        )

    derived_targets = host_capacity_slugs(runnability)
    if target_identity(targets) != target_identity(derived_targets):
        raise sr.PrerequisiteContractError(
            "shrink targets do not equal the complete host-capacity runnability set"
        )
    if (
        artifact.get("runnability_name") != selected_runnability_path.name
        or not selected_runnability_path.is_absolute()
    ):
        raise sr.PrerequisiteContractError(
            "shrink provenance runnability name does not match its external authority"
        )
    if reduced_output_root is None:
        raise sr.PrerequisiteContractError(
            "shrink provenance reduced-output root is required"
        )
    root_identity = validate_external_reduced_output_root(reduced_output_root)
    recorded_provenance_name = artifact.get("provenance_output_name")
    selected_provenance_path = (
        _external_output_path(root_identity, str(recorded_provenance_name))
        if provenance_path is None else Path(provenance_path)
    )
    if (
        not isinstance(recorded_provenance_name, str)
        or selected_provenance_path.absolute()
        != _external_output_path(root_identity, recorded_provenance_name)
    ):
        raise sr.PrerequisiteContractError(
            "shrink provenance output path does not match the loaded authority"
        )
    shrink_contract = build_shrink_contract(
        runnability_sha256=runnability_sha256,
        runnability=runnability,
        probe_contract=probe_contract,
        targets=targets,
        runnability_path=selected_runnability_path,
        reduced_output_root=reduced_output_root,
        provenance_output_path=selected_provenance_path,
        owner_mode="verify",
    )
    _validate_completed_shrink_against_contract(
        artifact, runnability=runnability, shrink_contract=shrink_contract,
        reduced_output_root_identity=root_identity,
    )


def load_completed_shrink_provenance(
    path: Path, *, runnability: dict[str, Any], runnability_sha256: str,
    probe_contract: dict[str, Any], targets: list[dict[str, Any]],
    runnability_path: Path | None = None,
    reduced_output_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load exact shrink bytes and enforce the complete current contract."""
    if reduced_output_root is None:
        raise sr.PrerequisiteContractError(
            "reduced output root is required for shrink provenance load"
        )
    root_identity = validate_external_reduced_output_root(reduced_output_root)
    if path.absolute() != _external_output_path(root_identity, path.name):
        raise sr.PrerequisiteContractError(
            "shrink provenance path is not a direct child of its owner root"
        )
    try:
        payload = _read_bound_direct_child(root_identity, path.name)
        artifact = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise sr.PrerequisiteContractError(
            f"completed shrink provenance is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(artifact, dict):
        raise sr.PrerequisiteContractError(
            "completed shrink provenance root is not an object"
        )
    validate_completed_shrink_provenance(
        artifact, runnability=runnability,
        runnability_sha256=runnability_sha256,
        probe_contract=probe_contract, targets=targets,
        runnability_path=runnability_path,
        provenance_path=path,
        reduced_output_root=reduced_output_root,
    )
    return artifact, _sha256(payload)


def _external_record_names(record: Mapping[str, Any]) -> set[str]:
    names = {
        _external_variant_name(record, row["variant_id"])
        for row in record.get("row_plans", [])
        if isinstance(row, dict) and row.get("planning_status") == "proxy_planned"
    }
    if (
        record.get("classification") == CLASS_REDUCED
        or record.get("cohort_execution_evidence") is not None
    ):
        names.add(_external_cohort_name(record, str(record.get("cohort_id"))))
    reduced_definition_ref = record.get("reduced_definition_ref")
    if isinstance(reduced_definition_ref, str):
        names.add(reduced_definition_ref)
    return names


def _expected_external_namespace(
    records: Mapping[str, Mapping[str, Any]], *, terminal_name: str,
) -> set[str]:
    names = {EXTERNAL_OWNER_MARKER_NAME, terminal_name}
    for record in records.values():
        names.update(_external_record_names(record))
    return names


def _external_child_manifest(
    records: Mapping[str, Mapping[str, Any]],
    root_identity: Mapping[str, Any],
) -> list[dict[str, str]]:
    names: set[str] = set()
    for record in records.values():
        names.update(_external_record_names(record))
    return [
        {
            "name": name,
            "sha256": _sha256(_read_bound_direct_child(root_identity, name)),
        }
        for name in sorted(names)
    ]


def apply_all(
    out: Path, *, slugs: set[str] | None, timeout: float,
    runnability_path: Path,
    reduced_output_root: Path,
) -> dict[str, Any]:
    if slugs is not None:
        raise sr.PrerequisiteContractError(
            "partial shrink mode is forbidden for the canonical provenance authority"
        )
    if timeout < sr.MIN_OUTER_TIMEOUT_SECONDS:
        raise sr.PrerequisiteContractError(
            f"shrink outer timeout must be >= {sr.MIN_OUTER_TIMEOUT_SECONDS:g}s"
        )
    fingerprint = sr.native_env_fingerprint()
    all_cells = sr.in_scope_cells()
    probe_contract = sr.build_probe_contract(
        native_env_hash=fingerprint["hash"], outer_timeout=timeout,
        target_cells=all_cells,
    )
    root_identity = validate_external_reduced_output_root(reduced_output_root)
    if out.absolute() != _external_output_path(root_identity, out.name):
        raise sr.PrerequisiteContractError(
            "shrink --out must be a direct child of --reduced-output-root"
        )
    runnability, runnability_sha = sr.load_completed_runnability(
        runnability_path, expected_contract=probe_contract,
        require_external_owner=True,
        external_campaign_root=runnability_path.parent,
    )
    targets = host_capacity_slugs(runnability)
    shrink_contract = build_shrink_contract(
        runnability_sha256=runnability_sha, runnability=runnability,
        probe_contract=probe_contract, targets=targets,
        runnability_path=runnability_path,
        reduced_output_root=reduced_output_root,
        provenance_output_path=out,
    )
    root_identity = shrink_contract["_reduced_output_root_identity"]
    checkpoint_name = f"{out.name}.partial"
    checkpoint = _external_output_path(root_identity, checkpoint_name)

    names = _bound_root_names(root_identity)
    if out.name in names:
        try:
            artifact = json.loads(_read_bound_direct_child(root_identity, out.name))
            if not isinstance(artifact, dict):
                raise TypeError("final shrink root is not an object")
            validate_completed_shrink_provenance(
                artifact, runnability=runnability,
                runnability_sha256=runnability_sha,
                probe_contract=probe_contract, targets=targets,
                runnability_path=runnability_path, provenance_path=out,
                reduced_output_root=reduced_output_root,
            )
            final_records = {row["slug"]: row for row in artifact["cells"]}
            expected = _expected_external_namespace(
                final_records, terminal_name=out.name
            )
            if checkpoint_name in names:
                expected.add(checkpoint_name)
            if names != expected:
                raise sr.PrerequisiteContractError(
                    "completed shrink namespace contains unexpected files"
                )
            if checkpoint_name in names:
                sr.unlink_external_child(root_identity, checkpoint_name)
            return artifact
        except (
            OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            sr.PrerequisiteContractError,
        ) as exc:
            raise sr.PrerequisiteContractError(
                f"refusing to overwrite stale/malformed shrink final: {exc}"
            ) from exc

    prior: dict[str, dict[str, Any]] = {}
    if checkpoint_name in names:
        try:
            existing = json.loads(_read_bound_direct_child(
                root_identity, checkpoint_name, require_readonly=False
            ))
            prior = _resumable_checkpoint_records(
                existing,
                shrink_contract=shrink_contract,
                reduced_output_root_identity=root_identity,
                runnability=runnability,
            )
        except (
            OSError, json.JSONDecodeError, sr.PrerequisiteContractError,
        ) as exc:
            raise sr.PrerequisiteContractError(
                f"shrink checkpoint is unreadable and will not be overwritten: {exc}"
            ) from exc
        expected = _expected_external_namespace(
            prior, terminal_name=checkpoint_name
        )
        if names != expected:
            raise sr.PrerequisiteContractError(
                "shrink resume namespace does not equal checkpoint-declared files: "
                f"expected={sorted(expected)} observed={sorted(names)}"
            )
    elif names != {EXTERNAL_OWNER_MARKER_NAME}:
        raise sr.PrerequisiteContractError(
            f"fresh shrink namespace contains unexpected files: {sorted(names)}"
        )

    target_slugs = {cell["slug"] for cell in targets}
    records = {slug: row for slug, row in prior.items() if slug in target_slugs}
    for cell in targets:
        if cell["slug"] in records:
            continue
        record = apply_one(
            cell, timeout, probe_contract,
            reduced_output_root_identity=root_identity,
        )
        records[cell["slug"]] = record
        _write(
            checkpoint, fingerprint, records, runnability, shrink_contract,
            publication_state="checkpoint",
        )
        expected = _expected_external_namespace(
            records, terminal_name=checkpoint_name
        )
        if _bound_root_names(root_identity) != expected:
            raise sr.PrerequisiteContractError(
                "shrink checkpoint namespace failed exact closure"
            )
        print(f"shrink {cell['category']}/{cell['slug']}: "
              f"{record['classification']}", flush=True)
    artifact = _write(
        out, fingerprint, records, runnability, shrink_contract,
        publication_state="complete",
    )
    if checkpoint_name in _bound_root_names(root_identity):
        sr.unlink_external_child(root_identity, checkpoint_name)
    if _bound_root_names(root_identity) != _expected_external_namespace(
        records, terminal_name=out.name
    ):
        raise sr.PrerequisiteContractError(
            "completed shrink namespace failed exact closure"
        )
    return artifact


def prior_host_capacity_reconciliation(
        runnability: dict | None,
        records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    entries = json.loads(DEFERRED_JSON.read_text()).get("entries", [])
    status_by_slug = {cell["slug"]: cell.get("status")
                      for cell in (runnability or {}).get("cells", [])}
    output = []
    for entry in entries:
        slug = entry["slug"]
        native_status = status_by_slug.get(slug)
        processed = slug in records and native_status == sr.HOST_CAPACITY_OOM
        output.append({
            "slug": slug,
            "category": entry.get("category"),
            "language": entry.get("language"),
            "prior_status": entry.get("status"),
            "native_sweep_status": native_status,
            "disposition": "shrunk" if processed else "retyped_by_native_sweep",
            "shrink_classification": (
                records[slug]["classification"] if processed else None),
        })
    return output


def _write(out: Path, fingerprint: dict[str, Any],
           records: dict[str, dict[str, Any]],
           runnability: dict | None,
           shrink_contract: dict[str, Any], *,
           publication_state: str) -> dict[str, Any]:
    cells = [records[key] for key in sorted(records)]
    root_identity = shrink_contract["_reduced_output_root_identity"]
    artifact = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": fingerprint["hash"],
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "publication_state": publication_state,
        **{
            key: value for key, value in shrink_contract.items()
            if not key.startswith("_")
        },
        "external_child_manifest": _external_child_manifest(
            records, root_identity
        ),
        "cells": cells,
        "prior_host_capacity_reconciliation":
            prior_host_capacity_reconciliation(runnability, records),
    }
    if publication_state == "complete":
        target_slugs = {row["slug"] for row in shrink_contract["targets"]}
        if set(records) != target_slugs:
            raise sr.PrerequisiteContractError(
                "refusing partial canonical shrink publication: "
                f"missing={sorted(target_slugs - set(records))} "
                f"extra={sorted(set(records) - target_slugs)}"
            )
        _validate_completed_shrink_against_contract(
            artifact, runnability=runnability,
            shrink_contract=shrink_contract,
            reduced_output_root_identity=shrink_contract[
                "_reduced_output_root_identity"
            ],
        )
    payload = canonical_json_bytes(artifact)
    if publication_state == "complete":
        sr.publish_external_child(root_identity, out.name, payload)
    else:
        sr.atomic_write_external_checkpoint(root_identity, out.name, payload)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--runnability", required=True, type=Path)
    parser.add_argument("--reduced-output-root", required=True, type=Path)
    parser.add_argument(
        "--outer-timeout", "--timeout", dest="outer_timeout", type=float,
        default=sr.DEFAULT_OUTER_TIMEOUT_SECONDS,
        help="outer process timeout; must be >= 2000 seconds",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    try:
        artifact = apply_all(
            out, slugs=None, timeout=args.outer_timeout,
            runnability_path=args.runnability,
            reduced_output_root=args.reduced_output_root,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"sol_shrink_apply: {exc}", file=sys.stderr)
        return 2
    by_classification = Counter(cell["classification"] for cell in artifact["cells"])
    print(f"sol shrink: {len(artifact['cells'])} host-capacity cells -> "
          f"{dict(by_classification)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
