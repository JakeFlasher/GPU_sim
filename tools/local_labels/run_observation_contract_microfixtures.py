#!/usr/bin/env python3
"""Live observation-contract microfixture sweep (sm_86 quarantine).

Pins, per replay-count metric, the counting semantic the replay-count
producer must implement — BEFORE any corpus scoring. Each microfixture
kernel's memory behavior is an exact closed-form function of the launch
geometry, so every (fixture, metric) pair carries one or more CANDIDATE
designed counts (named semantic models). The discovery rule per metric:

- the surviving semantic is the unique candidate whose designed count
  matches the live observed count on EVERY fixture that constrains the
  metric;
- zero survivors  -> the metric is typed ``contract_mismatch``
  (fail-closed: the validator refuses to score it);
- multiple survivors -> ``contract_underdetermined`` (fail-closed: the
  fixture set cannot separate the candidates; add fixtures).

Counts are admitted only when integer-stable across repeats (every
repeat integral and bitwise-equal), mirroring the count-label admission.
Surface ATOM/RED have no CUDA C++ expressible fixture; their entries are
typed ``no_expressible_fixture`` explicitly (the validator skips them
fail-closed rather than scoring them vacuously).

Outputs land ONLY under ``data/local_labels`` (Rule-1 quarantine). The
binary is a host-local build artifact (never committed); the artifact
records source+binary sha256 and the full device/driver/NCU identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_count_labels import (  # noqa: E402
    QuarantinePathError,
    assert_quarantine_path,
    assert_targets_probe_available,
    count_metric_identifiers,
)

FIXTURE_SOURCE = HERE / "microfixtures" / "obs_microfixtures.cu"
FIXTURE_BINARY = HERE / "microfixtures" / "build" / "obs_microfixtures"
DEFAULT_CONTRACT_OUT = "data/local_labels/observation_contract_sm_86.json"
DEFAULT_REPEATS = 5

# Launch geometry — MUST mirror obs_microfixtures.cu.
BLOCKS = 4
TPB = 128
THREADS = BLOCKS * TPB          # 512
WARPS = THREADS // 32           # 16
QUADS = THREADS // 4            # 128
INST_UNROLL = 64

_M = {
    "rollup": "l1tex__t_requests.sum",
    "g_atom": "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "g_ld": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "g_red": "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "g_st": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l_ld": "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
    "l_st": "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum",
    "s_atom": "l1tex__t_requests_pipe_tex_mem_surface_op_atom.sum",
    "s_ld": "l1tex__t_requests_pipe_tex_mem_surface_op_ld.sum",
    "s_red": "l1tex__t_requests_pipe_tex_mem_surface_op_red.sum",
    "s_st": "l1tex__t_requests_pipe_tex_mem_surface_op_st.sum",
    "tex": "l1tex__t_requests_pipe_tex_mem_texture.sum",
    "sec_ld": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "sec_st": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "inst": "smsp__inst_executed.sum",
}

# Pipe metrics whose sum the rollup-scope discovery compares against.
ROLLUP_PIPE_KEYS = (
    "g_atom", "g_ld", "g_red", "g_st", "l_ld", "l_st",
    "s_atom", "s_ld", "s_red", "s_st", "tex",
)

# Metrics with no CUDA-C++-expressible fixture on this corpus.
NO_EXPRESSIBLE_FIXTURE = ("s_atom", "s_red")

# ---------------------------------------------------------------------------
# Fixture table. Per fixture: binary args and, per metric key, the
# candidate designed counts {model_name: value}. A metric absent from a
# fixture's design map is UNCONSTRAINED there (e.g. inst_executed in
# memory fixtures). Zero-valued entries are real constraints: they pin
# cross-pipe non-contamination.
# ---------------------------------------------------------------------------


def _zeros(*keys: str) -> dict[str, dict[str, int]]:
    return {k: {"untouched_pipe_zero": 0} for k in keys}


FIXTURES: list[dict] = [
    {
        "name": "fx_ld32",
        "args": ["fx_ld32"],
        "design": {
            "g_ld": {"per_warp_inst": WARPS, "per_active_thread": THREADS},
            "g_st": {"per_warp_inst": WARPS, "per_active_thread": THREADS},
            "sec_ld": {"touched_32B_sectors": WARPS * 4},
            "sec_st": {"touched_32B_sectors": WARPS * 4},
            **_zeros("g_atom", "g_red", "l_ld", "l_st",
                     "s_atom", "s_ld", "s_red", "s_st", "tex"),
        },
    },
    {
        "name": "fx_ld64",
        "args": ["fx_ld64"],
        "design": {
            "g_ld": {"per_warp_inst": WARPS, "per_active_thread": THREADS,
                     "per_128B_wavefront": WARPS * 2},
            "g_st": {"per_warp_inst": WARPS, "per_active_thread": THREADS,
                     "per_128B_wavefront": WARPS * 2},
            "sec_ld": {"touched_32B_sectors": WARPS * 8},
            "sec_st": {"touched_32B_sectors": WARPS * 8},
            **_zeros("g_atom", "g_red", "l_ld", "l_st",
                     "s_atom", "s_ld", "s_red", "s_st", "tex"),
        },
    },
    {
        "name": "fx_ld128",
        "args": ["fx_ld128"],
        "design": {
            "g_ld": {"per_warp_inst": WARPS, "per_active_thread": THREADS,
                     "per_128B_wavefront": WARPS * 4},
            "g_st": {"per_warp_inst": WARPS, "per_active_thread": THREADS,
                     "per_128B_wavefront": WARPS * 4},
            "sec_ld": {"touched_32B_sectors": WARPS * 16},
            "sec_st": {"touched_32B_sectors": WARPS * 16},
            **_zeros("g_atom", "g_red", "l_ld", "l_st",
                     "s_atom", "s_ld", "s_red", "s_st", "tex"),
        },
    },
    {
        "name": "fx_misaligned",
        "args": ["fx_misaligned"],
        "design": {
            "g_ld": {"per_warp_inst": WARPS},
            "g_st": {"per_warp_inst": WARPS},
            "sec_st": {"touched_32B_sectors": WARPS * 4},
            **_zeros("g_atom", "g_red", "l_ld", "l_st",
                     "s_atom", "s_ld", "s_red", "s_st", "tex"),
        },
        # The 4B-offset warp load naively touches 5 sectors per warp
        # (designed 80); the live counter does NOT follow touched-sector
        # arithmetic under sub-sector misalignment. This probe records
        # the anomaly and SCOPE-RESTRICTS the pinned sector semantic to
        # 128B-aligned warp footprints (misaligned cells must not be
        # scored against the touched-sector model).
        "scope_probes": {
            "sec_ld": {"naive_touched_32B_sectors": WARPS * 5},
        },
    },
    {
        "name": "fx_lane_subset_w1",
        "args": ["fx_lane_subset", "1"],
        "design": {
            # One active lane per warp: a request still forms per warp
            # instruction; only the single touched sector counts.
            "g_ld": {"per_warp_inst": WARPS, "per_active_thread": WARPS},
            "g_st": {"per_warp_inst": WARPS, "per_active_thread": WARPS},
            "sec_ld": {"touched_32B_sectors": WARPS * 1},
            "sec_st": {"touched_32B_sectors": WARPS * 1},
            **_zeros("g_atom", "g_red", "l_ld", "l_st",
                     "s_atom", "s_ld", "s_red", "s_st", "tex"),
        },
    },
    {
        "name": "fx_lane_subset_w32",
        "args": ["fx_lane_subset", "32"],
        "design": {
            "g_ld": {"per_warp_inst": WARPS, "per_active_thread": THREADS},
            "g_st": {"per_warp_inst": WARPS, "per_active_thread": THREADS},
            "sec_ld": {"touched_32B_sectors": WARPS * 4},
            "sec_st": {"touched_32B_sectors": WARPS * 4},
            **_zeros("g_atom", "g_red", "l_ld", "l_st",
                     "s_atom", "s_ld", "s_red", "s_st", "tex"),
        },
    },
    {
        "name": "fx_atom_return",
        "args": ["fx_atom_return"],
        "sass_witness": {"require": ["ATOMG"], "forbid": ["RED."]},
        "design": {
            "g_atom": {"per_warp_inst": WARPS, "per_active_thread": THREADS},
            "g_st": {"per_warp_inst": WARPS},  # the sink store
            "sec_st": {"touched_32B_sectors": WARPS * 4},
            **_zeros("g_ld", "g_red", "l_ld", "l_st",
                     "s_atom", "s_ld", "s_red", "s_st", "tex", "sec_ld"),
        },
    },
    {
        "name": "fx_red_noreturn",
        "args": ["fx_red_noreturn"],
        "sass_witness": {"require": ["RED."], "forbid": ["ATOMG"]},
        "design": {
            "g_red": {"per_warp_inst": WARPS, "per_active_thread": THREADS},
            **_zeros("g_ld", "g_st", "g_atom", "l_ld", "l_st",
                     "s_atom", "s_ld", "s_red", "s_st", "tex",
                     "sec_ld", "sec_st"),
        },
    },
    {
        "name": "fx_local_rw",
        "args": ["fx_local_rw"],
        "sass_witness": {"require": ["STL", "LDL"]},
        "design": {
            # 32b local elements: per_warp_inst and per_32b_word_split
            # coincide here (one word per access); the wide fixtures
            # separate them.
            "l_st": {"per_warp_inst": WARPS, "per_active_thread": THREADS,
                     "per_32b_word_split": WARPS},
            "l_ld": {"per_warp_inst": WARPS, "per_active_thread": THREADS,
                     "per_32b_word_split": WARPS},
            "g_ld": {"per_warp_inst": WARPS},
            "g_st": {"per_warp_inst": WARPS},
            "sec_ld": {"touched_32B_sectors": WARPS * 4},
            "sec_st": {"touched_32B_sectors": WARPS * 4},
            **_zeros("g_atom", "g_red", "s_atom", "s_ld", "s_red",
                     "s_st", "tex"),
        },
    },
    {
        "name": "fx_local_wide64",
        "args": ["fx_local_wide64"],
        "sass_witness": {"require": ["STL", "LDL"]},
        "design": {
            # Does the local pipe split a 64-bit access into multiple
            # requests? The 32b fixture cannot distinguish these.
            "l_st": {"per_warp_inst": WARPS,
                     "per_32b_word_split": WARPS * 2},
            "l_ld": {"per_warp_inst": WARPS,
                     "per_32b_word_split": WARPS * 2},
            "g_ld": {"per_warp_inst": WARPS},
            "g_st": {"per_warp_inst": WARPS},
            "sec_ld": {"touched_32B_sectors": WARPS * 8},
            "sec_st": {"touched_32B_sectors": WARPS * 8},
            **_zeros("g_atom", "g_red", "s_atom", "s_ld", "s_red",
                     "s_st", "tex"),
        },
    },
    {
        "name": "fx_local_wide128",
        "args": ["fx_local_wide128"],
        "sass_witness": {"require": ["STL", "LDL"]},
        "design": {
            "l_st": {"per_warp_inst": WARPS,
                     "per_32b_word_split": WARPS * 4,
                     "per_64b_half_split": WARPS * 2},
            "l_ld": {"per_warp_inst": WARPS,
                     "per_32b_word_split": WARPS * 4,
                     "per_64b_half_split": WARPS * 2},
            "g_ld": {"per_warp_inst": WARPS},
            "g_st": {"per_warp_inst": WARPS},
            "sec_ld": {"touched_32B_sectors": WARPS * 16},
            "sec_st": {"touched_32B_sectors": WARPS * 16},
            **_zeros("g_atom", "g_red", "s_atom", "s_ld", "s_red",
                     "s_st", "tex"),
        },
    },
    {
        "name": "fx_tex",
        "args": ["fx_tex"],
        "design": {
            "tex": {"per_warp_inst": WARPS, "per_quad": QUADS,
                    "per_active_thread": THREADS,
                    "four_wavefronts_per_warp_inst": WARPS * 4},
            "g_st": {"per_warp_inst": WARPS},
            "sec_st": {"touched_32B_sectors": WARPS * 4},
            **_zeros("g_ld", "g_atom", "g_red", "l_ld", "l_st",
                     "s_atom", "s_ld", "s_red", "s_st", "sec_ld"),
        },
    },
    {
        "name": "fx_surf_ld",
        "args": ["fx_surf_ld"],
        "sass_witness": {"require": ["SULD"]},
        "design": {
            "s_ld": {"per_warp_inst": WARPS, "per_quad": QUADS,
                     "per_active_thread": THREADS,
                     "four_wavefronts_per_warp_inst": WARPS * 4},
            "g_st": {"per_warp_inst": WARPS},
            "sec_st": {"touched_32B_sectors": WARPS * 4},
            **_zeros("g_ld", "g_atom", "g_red", "l_ld", "l_st",
                     "s_atom", "s_red", "s_st", "tex", "sec_ld"),
        },
    },
    {
        "name": "fx_surf_st",
        "args": ["fx_surf_st"],
        "sass_witness": {"require": ["SUST"]},
        "design": {
            "s_st": {"per_warp_inst": WARPS, "per_quad": QUADS,
                     "per_active_thread": THREADS,
                     "four_wavefronts_per_warp_inst": WARPS * 4},
            "g_ld": {"per_warp_inst": WARPS},
            "sec_ld": {"touched_32B_sectors": WARPS * 4},
            **_zeros("g_st", "g_atom", "g_red", "l_ld", "l_st",
                     "s_atom", "s_ld", "s_red", "tex", "sec_st"),
        },
    },
    # Instruction-semantics triple: same SASS, runtime W in {32,16,1}.
    {"name": "fx_inst_pred_w32", "args": ["fx_inst_pred", "32"], "design": {}},
    {"name": "fx_inst_pred_w16", "args": ["fx_inst_pred", "16"], "design": {}},
    {"name": "fx_inst_pred_w1", "args": ["fx_inst_pred", "1"], "design": {}},
]

# inst_executed delta models over the (w32, w16, w1) triple:
#   issued_warp + divergence reconvergence -> the guarded body issues
#     per WARP regardless of the active mask; the w=32 launch takes the
#     UNIFORM branch path (one fewer warp instruction than the divergent
#     w=16/w=1 paths), so deltas are (-WARPS, 0). The (w16 - w1) == 0
#     arm is the strong issued-warp witness: active-lane counting would
#     put ~WARPS*INST_UNROLL*15 instructions between them.
#   active_lane_sum -> each issued instruction contributes its active
#     lane count: deltas proportional to the masked lanes.
INST_DELTA_MODELS = {
    "issued_warp_instructions_with_divergence_reconvergence": (
        -WARPS, 0,
    ),
    "issued_warp_instructions_uniform_paths": (0, 0),
    "active_lane_sum": (
        WARPS * INST_UNROLL * (32 - 16),
        WARPS * INST_UNROLL * (16 - 1),
    ),
}


class FixtureCountUnstableError(RuntimeError):
    typed_reason = "fixture_count_unstable"


class SassWitnessError(RuntimeError):
    typed_reason = "fixture_sass_witness_failed"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixtures(nvcc: str = "nvcc", arch: str = "sm_86") -> Path:
    FIXTURE_BINARY.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [nvcc, "-O3", f"-arch={arch}", str(FIXTURE_SOURCE),
         "-o", str(FIXTURE_BINARY)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed: {proc.stderr[-500:]}")
    return FIXTURE_BINARY


def sass_witness(binary: Path) -> dict[str, dict[str, int]]:
    """Per-kernel opcode-presence counts from static SASS (cuobjdump).
    The witness proves each fixture exercises the op class it claims
    (e.g. the no-return atomic really lowered to RED, not ATOMG)."""
    proc = subprocess.run(
        ["cuobjdump", "--dump-sass", str(binary)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {proc.stderr[-300:]}")
    out: dict[str, dict[str, int]] = {}
    current = None
    opcodes = ("ATOMG", "RED.", "STL", "LDL", "SULD", "SUST",
               "TLD", "TEX", "LDG", "STG")
    for line in proc.stdout.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            current = m.group(1)
            out[current] = {op: 0 for op in opcodes}
            continue
        if current is None:
            continue
        for op in opcodes:
            if op in line:
                out[current][op] += 1
    return out


def check_sass_witness(
    fixture: dict, witness_by_kernel: dict[str, dict[str, int]],
) -> dict:
    """Match the fixture's required/forbidden opcodes against the kernel
    whose mangled name embeds the fixture's kernel symbol."""
    spec = fixture.get("sass_witness")
    kernel_symbol = fixture["args"][0]
    matches = [
        (name, ops) for name, ops in witness_by_kernel.items()
        if kernel_symbol in name
    ]
    if len(matches) != 1:
        raise SassWitnessError(
            f"{fixture['name']}: expected exactly one kernel matching "
            f"{kernel_symbol!r} in SASS, found {len(matches)}"
        )
    name, ops = matches[0]
    if spec:
        for op in spec.get("require", []):
            if ops.get(op, 0) == 0:
                raise SassWitnessError(
                    f"{fixture['name']}: required opcode {op} absent from "
                    f"{name} SASS"
                )
        for op in spec.get("forbid", []):
            if ops.get(op, 0) != 0:
                raise SassWitnessError(
                    f"{fixture['name']}: forbidden opcode {op} present in "
                    f"{name} SASS"
                )
    return {"kernel": name, "opcode_counts": ops}


def _ncu_capture_once_filtered(
    ncu: str, binary: Path, app_args: list[str],
    metrics: tuple[str, ...],
) -> dict[str, float]:
    """One read-only NCU capture scoped to the fixture kernels via the
    NCU-side ``--kernel-name`` filter (host memset/driver launches are
    excluded). CSV to stdout only — no .ncu-rep is ever written."""
    import csv
    import io

    proc = subprocess.run(
        [
            ncu, "--metrics", ",".join(metrics),
            "--kernel-name", "regex:fx_",
            "--launch-count", "1", "--csv", str(binary), *app_args,
        ],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ncu rc={proc.returncode}: {proc.stderr[-400:]!r}")
    rows = [ln for ln in proc.stdout.splitlines() if ln.startswith('"')]
    reader = csv.DictReader(io.StringIO("\n".join(rows)))
    out: dict[str, float] = {}
    for row in reader:
        name = row.get("Metric Name")
        value = row.get("Metric Value")
        if name in metrics and value:
            out[name] = float(value.replace(",", ""))
    return out


def capture_fixture(
    ncu: str, binary: Path, fixture: dict, metrics: tuple[str, ...],
    repeats: int,
) -> dict[str, float]:
    """Integer-stability-admitted counts for one fixture."""
    raw: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(repeats):
        captured = _ncu_capture_once_filtered(
            ncu, binary, list(fixture["args"]), metrics,
        )
        for m in metrics:
            if m in captured:
                raw[m].append(captured[m])
    stable: dict[str, float] = {}
    for m, values in raw.items():
        if len(values) != repeats:
            raise FixtureCountUnstableError(
                f"{fixture['name']}/{m}: {len(values)}/{repeats} repeats"
            )
        if any(v != int(v) for v in values) or max(values) != min(values):
            raise FixtureCountUnstableError(
                f"{fixture['name']}/{m}: repeats not integer-stable: "
                f"{values}"
            )
        stable[m] = values[0]
    return stable


def discover_semantics(
    observed_by_fixture: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """Per metric: the unique candidate model matching every
    constraining fixture, or a typed fail-closed status."""
    out: dict[str, dict] = {}
    for key, metric in _M.items():
        if key in NO_EXPRESSIBLE_FIXTURE:
            out[metric] = {
                "status": "no_expressible_fixture",
                "note": (
                    "CUDA C++ exposes no surface atomic/reduction API; "
                    "the validator must skip this metric fail-closed "
                    "rather than score it vacuously."
                ),
                "evidence": [
                    {
                        "fixture": fx["name"],
                        "designed": {"untouched_pipe_zero": 0},
                        "observed": observed_by_fixture[fx["name"]][metric],
                    }
                    for fx in FIXTURES
                    if key in fx.get("design", {})
                ],
            }
            continue
        if key == "inst":
            out[metric] = _discover_inst(observed_by_fixture)
            continue
        if key == "rollup":
            out[metric] = _discover_rollup(observed_by_fixture)
            continue
        evidence = []
        contamination = []
        survivors: set[str] | None = None
        scope_anomalies = []
        for fx in FIXTURES:
            obs = observed_by_fixture[fx["name"]][metric]
            probe = fx.get("scope_probes", {}).get(key)
            if probe is not None:
                # Scope probe: documents where the pinned semantic's
                # naive arithmetic breaks; NOT part of the candidate
                # intersection. A recorded anomaly scope-restricts the
                # pinned semantic fail-closed.
                anomaly = {
                    "fixture": fx["name"],
                    "naive_designed": probe,
                    "observed": obs,
                    "anomalous": all(obs != v for v in probe.values()),
                }
                scope_anomalies.append(anomaly)
                continue
            models = fx.get("design", {}).get(key)
            if not models:
                continue
            if set(models) == {"untouched_pipe_zero"}:
                # Contamination gate: the fixture does not exercise this
                # pipe, so any non-zero count poisons the fixture set.
                contamination.append({
                    "fixture": fx["name"],
                    "observed": obs,
                    "clean": obs == 0,
                })
                continue
            matching = {name for name, v in models.items() if obs == v}
            evidence.append({
                "fixture": fx["name"],
                "designed": models,
                "observed": obs,
                "matching_models": sorted(matching),
            })
            survivors = matching if survivors is None else (
                survivors & matching
            )
        contaminated = any(not c["clean"] for c in contamination)
        survivors = survivors or set()
        if contaminated or not evidence:
            status, semantic = "contract_mismatch", None
        elif len(survivors) == 1:
            status, semantic = "pinned", sorted(survivors)[0]
        elif not survivors:
            status, semantic = "contract_mismatch", None
        else:
            status, semantic = "contract_underdetermined", None
        entry: dict = {
            "status": status,
            "semantic": semantic,
            "evidence": evidence,
            "contamination_gates": contamination,
        }
        if scope_anomalies:
            entry["scope_probes"] = scope_anomalies
            if any(a["anomalous"] for a in scope_anomalies):
                entry["semantic_scope"] = (
                    "aligned_128B_warp_footprints_only"
                )
                entry["scope_note"] = (
                    "The pinned touched-sector arithmetic does NOT "
                    "reproduce the live counter under sub-sector "
                    "misalignment (see scope_probes); cells whose "
                    "traces contain misaligned warp footprints must "
                    "be typed, not scored, for this metric."
                )
        out[metric] = entry
    return out


def _discover_inst(observed_by_fixture: dict[str, dict[str, float]]) -> dict:
    metric = _M["inst"]
    c32 = observed_by_fixture["fx_inst_pred_w32"][metric]
    c16 = observed_by_fixture["fx_inst_pred_w16"][metric]
    c1 = observed_by_fixture["fx_inst_pred_w1"][metric]
    deltas = (c32 - c16, c16 - c1)
    matching = {
        name for name, d in INST_DELTA_MODELS.items() if deltas == d
    }
    if len(matching) == 1:
        status, semantic = "pinned", sorted(matching)[0]
    elif not matching:
        status, semantic = "contract_mismatch", None
    else:
        status, semantic = "contract_underdetermined", None
    return {
        "status": status,
        "semantic": semantic,
        "evidence": [{
            "fixture": "fx_inst_pred_w{32,16,1}",
            "designed": {
                name: list(d) for name, d in INST_DELTA_MODELS.items()
            },
            "observed_counts": [c32, c16, c1],
            "observed_deltas": list(deltas),
            "matching_models": sorted(matching),
        }],
    }


def _discover_rollup(observed_by_fixture: dict[str, dict[str, float]]) -> dict:
    metric = _M["rollup"]
    evidence = []
    residual_zero = True
    for fx in FIXTURES:
        obs = observed_by_fixture[fx["name"]]
        pipe_sum = sum(obs[_M[k]] for k in ROLLUP_PIPE_KEYS)
        residual = obs[metric] - pipe_sum
        evidence.append({
            "fixture": fx["name"],
            "rollup_observed": obs[metric],
            "pipe_sum_observed": pipe_sum,
            "residual": residual,
        })
        if residual != 0:
            residual_zero = False
    return {
        "status": "pinned" if residual_zero else "contract_mismatch",
        "semantic": (
            "sum_of_l1tex_pipe_request_metrics" if residual_zero else None
        ),
        "included_pipes": [_M[k] for k in ROLLUP_PIPE_KEYS],
        "scope_note": (
            "Scope pinned over the exercised op classes only: global "
            "ld/st/atom/red, local ld/st, surface ld/st, texture. A "
            "trace containing an op class outside this set must not "
            "score the rollup (rollup_unsupported)."
        ),
        "evidence": evidence,
    }


def collect_device_identity(ncu: str) -> dict:
    out: dict = {}
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            name, driver = [s.strip() for s in
                            smi.stdout.strip().split(",", 1)]
            out["device"] = name
            out["driver_version"] = driver
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        ver = subprocess.run([ncu, "--version"], capture_output=True,
                             text=True, timeout=60)
        m = re.search(r"Version (\S+)", ver.stdout)
        if m:
            out["ncu_version"] = m.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--contract-out", default=DEFAULT_CONTRACT_OUT)
    args = parser.parse_args(argv)

    contract_path = Path(args.contract_out)
    try:
        assert_quarantine_path(contract_path)
    except QuarantinePathError as exc:
        print(json.dumps({"typed_reason": exc.typed_reason,
                          "detail": str(exc)}), file=sys.stderr)
        return 3

    binary = build_fixtures(nvcc=args.nvcc, arch=args.arch)
    witness_by_kernel = sass_witness(binary)

    metrics = count_metric_identifiers()
    assert_targets_probe_available(metrics)

    observed_by_fixture: dict[str, dict[str, float]] = {}
    fixture_records = []
    for fx in FIXTURES:
        wit = check_sass_witness(fx, witness_by_kernel)
        stable = capture_fixture(args.ncu, binary, fx, metrics,
                                 args.repeats)
        observed_by_fixture[fx["name"]] = stable
        fixture_records.append({
            "name": fx["name"],
            "args": fx["args"],
            "sass_witness": wit,
            "observed": stable,
        })
        print(f"[obs_contract] {fx['name']}: captured "
              f"{len(stable)} metrics integer-stable", flush=True)

    metric_entries = discover_semantics(observed_by_fixture)

    n_pinned = sum(1 for e in metric_entries.values()
                   if e["status"] == "pinned")
    artifact = {
        "schema_version": 1,
        "artifact_kind": "observation_contract",
        "arch": args.arch,
        **collect_device_identity(args.ncu),
        "n_repeats": args.repeats,
        "geometry": {"blocks": BLOCKS, "threads_per_block": TPB,
                     "warps": WARPS, "inst_unroll": INST_UNROLL},
        "build": {
            "nvcc_arch": args.arch,
            "source_sha256": _sha256(FIXTURE_SOURCE),
            "binary_sha256": _sha256(binary),
        },
        "fixtures": fixture_records,
        "metrics": metric_entries,
        "summary": {
            "n_metrics": len(metric_entries),
            "n_pinned": n_pinned,
            "n_no_expressible_fixture": sum(
                1 for e in metric_entries.values()
                if e["status"] == "no_expressible_fixture"),
            "n_contract_mismatch": sum(
                1 for e in metric_entries.values()
                if e["status"] == "contract_mismatch"),
            "n_contract_underdetermined": sum(
                1 for e in metric_entries.values()
                if e["status"] == "contract_underdetermined"),
        },
    }
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[obs_contract] wrote {contract_path}: "
          f"{n_pinned}/{len(metric_entries)} metrics pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
