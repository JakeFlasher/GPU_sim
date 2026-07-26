"""Adversarial tests for the campaign chunk DONE/resume seal."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import campaign_done_manifest as dm  # noqa: E402
from tools.local_labels import campaign_exclusion_ledger as el  # noqa: E402
from tools.local_labels import capture_budget_route as br  # noqa: E402
from tools.local_labels import ncu_action_executor  # noqa: E402
from tools.local_labels import ncu_collection_receipt  # noqa: E402
from tools.local_labels import sol_definition_equivalence as sol_def  # noqa: E402
from tools.trace_sampling.counting_pass import BlockStat, CellStats  # noqa: E402

_SCHEMA = json.loads(
    (_REPO / "workloads" / "simbench_bw" / "schemas"
     / "campaign_done_manifest.schema.json").read_text()
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_done_native_environment_hash_is_derived_from_fingerprint_authority(
    tmp_path, monkeypatch,
):
    source = json.loads((_REPO / dm.SOL_NATIVE_ENV_PIN_PATH).read_text())
    assert dm.SOL_CERTIFIED_NATIVE_ENV_HASH == dm._canonical_sha256(
        source["components"]
    )
    planted = copy.deepcopy(source)
    planted["native_env_hash"] = "0" * 64
    path = tmp_path / "fingerprint.json"
    _write_json(path, planted)
    monkeypatch.setattr(dm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dm, "SOL_NATIVE_ENV_PIN_PATH", path.name)
    with pytest.raises(RuntimeError, match="native_env_hash is stale"):
        dm._load_certified_sol_native_env_hash()


def _sol_attestation_document(identity: dict, pin: dict) -> dict:
    certified = json.loads((_REPO / dm.SOL_NATIVE_ENV_PIN_PATH).read_text())
    unsigned = {
        "schema_version": dm.SOL_NATIVE_ENV_ATTESTATION_SCHEMA_VERSION,
        "kind": dm.SOL_NATIVE_ENV_ATTESTATION_KIND,
        "identity": identity,
        "certified_native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
        "observed_native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
        "certified_hash_equal": True,
        "degraded": False,
        "missing_components": [],
        "components": certified["components"],
        "components_sha256": dm._canonical_sha256(certified["components"]),
        "compiler_identity": {
            "requested": dm.SOL_CERTIFIED_NVCC_CCBIN,
            "resolved_realpath": "/usr/bin/x86_64-linux-gnu-g++-14",
            "executable_sha256": "e" * 64,
            "version_line": "g++-14 (GCC) 14.3.1 20260214",
        },
        "matrix_pin": pin,
    }
    return {
        **unsigned,
        "attestation_sha256": dm._canonical_sha256(unsigned),
    }


def _sol_env_attestation_case(tmp_path: Path, *, disposition: str = "admitted") -> dict:
    certified_path = _REPO / dm.SOL_NATIVE_ENV_PIN_PATH
    pin = {
        "path": dm.SOL_NATIVE_ENV_PIN_PATH,
        "sha256": _sha(certified_path),
        "native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
    }
    identity = {"roster_source": "sol_execbench", "slug": "problem_070"}
    attestation = _sol_attestation_document(identity, pin)
    output_root = tmp_path / "sol-capture"
    rel = "cells/sol_execbench__problem_070/sol_native_env_fingerprint.json"
    path = output_root / rel
    _write_json(path, attestation)
    row = {
        **identity,
        "authority_cell_sha256": "a" * 64,
        "disposition": disposition,
        "typed_reason": None if disposition == "admitted" else "trace_unsupported",
        "detail": None,
        "artifacts": [{
            "artifact_kind": "sol_capture_evidence",
            "relative_path": rel,
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }],
    }
    matrix = {
        "arch": "sm_86",
        "input_pins": {dm.SOL_NATIVE_ENV_PIN_SECTION: pin},
        "cells": [{
            "identity": identity,
            "disposition": "runnable",
            "authority_cell_sha256": "a" * 64,
            "invocation": {
                "adapter_contract": {
                    "native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
                    "environment": dict(dm.SOL_CERTIFIED_ENVIRONMENT),
                },
            },
        }],
    }
    return {
        "matrix": matrix,
        "manifest": {"output_root": str(output_root), "cell_results": [row]},
        "attestation": attestation,
        "attestation_path": path,
        "artifact": row["artifacts"][0],
    }


def _route_stats(n_blocks: int = 60) -> CellStats:
    blocks = []
    for block_id in range(n_blocks):
        block = BlockStat(
            block_id=block_id,
            sm_id=0,
            records=100,
            sector_accesses=10,
            serialized_bytes=4000,
            warps_per_block=1,
            max_warp_slot=0,
        )
        block.wave = block_id
        blocks.append(block)
    return CellStats(
        cell_id="cell", kernel_name="candidate_kernel",
        grid=(n_blocks, 1, 1), block=(32, 1, 1),
        n_blocks=n_blocks, total_records=n_blocks * 100,
        op_class_totals={1: n_blocks * 100}, blocks=blocks,
        per_sm_resident={0: 1}, per_sm_block_ids={0: list(range(n_blocks))},
        n_waves=n_blocks, atomic_records=0,
        atomic_top256_sector_share=0.0,
        per_wave_record_totals={block_id: 100 for block_id in range(n_blocks)},
        per_sm_block_counts={0: n_blocks}, stats_pass_version=2,
        serialized_bytes_exact=True,
    )


def _admitted_route_case(tmp_path: Path) -> dict:
    identity = {"roster_source": "portable_lcd", "slug": "cell"}
    stats = _route_stats()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dynamic_trace.pb").write_bytes(b"root-envelope-v1")
    (raw / "stats.csv").write_text(
        "kernel_id,total_insts,total_reported_insts\n"
        f"1,{stats.total_records},{stats.total_records}\n"
    )
    blocks = raw / "threadblocks/device_0/stream_0/kernel_1"
    blocks.mkdir(parents=True)
    for block_id in range(stats.n_blocks):
        (blocks / f"block_{block_id:04d}.pb").write_bytes(b"x" * 4000)
    raw_inventory, raw_bytes, raw_sha = br._tree_inventory(raw)
    budget = {
        "max_trace_bytes": 80_000,
        "max_wall_time_s": 600,
        "sampling_policy": "off",
    }
    invocation = {
        "adapter_kind": "direct_binary",
        "executable_sha256": "7" * 64,
        "trace_app_args": ["--timing_shape", "--warmup", "0", "--reps", "1"],
        "logical_workload": {"kind": "shape_point", "value": "timing_shape"},
        "trace_cardinality_contract": "runner_common_single_launch_v1",
    }
    args_hash = dm._canonical_sha256({
        "adapter_kind": invocation["adapter_kind"],
        "trace_app_args": invocation["trace_app_args"],
        "logical_workload": invocation["logical_workload"],
        "trace_cardinality_contract": invocation["trace_cardinality_contract"],
    })
    cell_dir = tmp_path / "capture/cells/portable_lcd__cell"
    trace_dir = cell_dir / "trace"
    cell_dir.mkdir(parents=True)
    gate = br.classify(stats, sample_size=br.MIN_SEGMENTS)

    def sidecar(comp, _block_manifest):
        return br.build_sidecar(
            cell_id="cell", kernel_name=stats.kernel_name,
            args_hash=args_hash, stats=stats, gate=gate, compression=comp,
            records_captured=stats.total_records, tracer_version=br.TRACER_VERSION,
            stats_pass_version=stats.stats_pass_version,
            method="lossless_compressed_full", reason="byte_budget",
        )

    br._build_compressed_bundle(raw, trace_dir, sidecar)
    trace_manifest = {
        "schema_version": 1,
        "identity": identity,
        "binary_sha256": "7" * 64,
        "tracer_so_sha256": "8" * 64,
        "max_trace_bytes": budget["max_trace_bytes"],
        "trace_tree_bytes": raw_bytes,
        "launch_count": 1,
        "threadblocks_pb_count": stats.n_blocks,
        "grid_threadblock_count": stats.n_blocks,
        "decoded_threadblock_count": stats.n_blocks,
        "decoded_record_count": stats.total_records,
        "zero_record_threadblock_count": 0,
        "stats_row_count": 1,
        "expected_records": stats.total_records,
        "admitted_records": stats.total_records,
        "drop_rate": 0.0,
    }
    trace_manifest_path = cell_dir / "trace_capture_manifest.json"
    _write_json(trace_manifest_path, trace_manifest)
    protocol = {
        "protocol_version": "v2.1",
        "source_commit": "1" * 40,
        "authority_cell_sha256": "2" * 64,
        "capture_matrix_sha256": "3" * 64,
        "preflight_sha256": "4" * 64,
        "pins_sha256": "5" * 64,
        "trace_budget_sha256": "6" * 64,
        "capture_manifest_sha256": _sha(trace_manifest_path),
        "binary_sha256": "7" * 64,
        "tracer_so_sha256": "8" * 64,
        "args_hash": args_hash,
    }
    full = br.FullTraceEvidence(
        stats=stats, inventory=raw_inventory, tree_sha256=raw_sha,
        tree_bytes=raw_bytes, root={"launch_count": 1},
        expected_records=stats.total_records, admitted_records=stats.total_records,
        coverage={
            "threadblocks_pb_count": stats.n_blocks,
            "decoded_record_count": stats.total_records,
        },
    )
    config = br.RouteConfig(
        selection_max_raw_bytes=40_000,
        staging_limit_bytes=1_000_000,
        gate_forbidden_policy="typed_unavailable",
    )
    route = br._result_record(
        identity=identity, cell_id="cell", budget=budget, config=config,
        protocol=protocol, full=full, output_dir=trace_dir,
        method="lossless_compressed_full", outcome="admitted", typed_reason=None,
    )
    route_path = cell_dir / "trace_budget_route.json"
    br.write_route_result(route, route_path)
    cell_result = {
        "schema_version": 1,
        "identity": identity,
        "authority_cell_sha256": "2" * 64,
        "disposition": "admitted",
        "typed_reason": None,
        "binary_sha256": "7" * 64,
        "capture_preflight_sha256": "4" * 64,
        "trace_route_outcome": "admitted",
        "trace_route_result_sha256": route["result_sha256"],
    }
    cell_result_path = cell_dir / "cell_capture_result.json"
    _write_json(cell_result_path, cell_result)

    def artifact(path: Path, kind: str) -> dict:
        return {
            "artifact_kind": kind,
            "relative_path": path.relative_to(tmp_path / "capture").as_posix(),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }

    row = {
        **identity,
        "authority_cell_sha256": "2" * 64,
        "disposition": "admitted",
        "typed_reason": None,
        "detail": None,
        "artifacts": [
            artifact(route_path, "trace_route_record"),
            artifact(trace_manifest_path, "trace_manifest"),
            artifact(cell_result_path, "cell_result"),
        ],
    }
    matrix = {
        "cells": [{
            "identity": identity,
            "authority_cell_sha256": "2" * 64,
            "disposition": "runnable",
            "invocation": invocation,
            "trace_budget": {
                "source_path": "workloads/simbench_bw/trace_budgets.toml",
                "source_sha256": "6" * 64,
                "status": "declared",
                **budget,
            },
        }],
        "input_pins": {
            "per_cell_trace_budgets": {
                "path": "workloads/simbench_bw/trace_budgets.toml",
                "sha256": "6" * 64,
            },
        },
    }
    manifest = {
        "protocol_version": "v2.1",
        "source_commit_pre": "1" * 40,
        "capture_matrix_sha256": "3" * 64,
        "preflight_sha256": "4" * 64,
        "pins_sha256": "5" * 64,
        "output_root": str(tmp_path / "capture"),
        "cell_results": [row],
    }
    preflight = {
        "checks": {"injection_runtime": {"tracer_so_sha256": "8" * 64}},
    }
    return {
        "row": row,
        "manifest": manifest,
        "matrix": matrix,
        "preflight": preflight,
        "route": route,
        "route_path": route_path,
        "trace_dir": trace_dir,
        "route_artifact": row["artifacts"][0],
    }


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    )
    return proc.stdout.strip()


@pytest.fixture
def case(tmp_path: Path) -> dict:
    """A clean tiny git repo plus external, fully hashed capture evidence."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "DONE test")
    _git(repo, "config", "user.email", "done@example.invalid")
    (repo / "README").write_text("source\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-qm", "source")
    head = _git(repo, "rev-parse", "HEAD")

    inputs = tmp_path / "inputs"
    pins_path = inputs / "pins.json"
    pins = {
        "arch": "sm_86",
        "measurement_protocol": {"version": "v2.1"},
        "toolchain": {
            "ncu_version_prefix": "2026.1.0",
            "driver_version": "610.43.02",
            "nvcc_release": "13.3",
            "nvbit_version": "1.8",
        },
    }
    _write_json(pins_path, pins)

    identities = [("portable_lcd", "cell_a"), ("simbench_bw", "cell_b")]
    authority_cells = [
        {
            "roster_source": source,
            "slug": slug,
            "resolution_status": "runnable",
            "binary": f"bin/{slug}",
        }
        for source, slug in identities
    ]
    authority_path = inputs / "authority.json"
    _write_json(authority_path, {
        "source_commit": head,
        "arch": "sm_86",
        "cells": authority_cells,
    })
    matrix_path = inputs / "matrix.json"
    compact_identities = [
        {"roster_source": source, "slug": slug}
        for source, slug in identities
    ]
    matrix = {
        "source_commit": head,
        "arch": "sm_86",
        "cells": [
            {
                "identity": identity,
                "disposition": "runnable",
                "authority_cell_sha256": dm.authority_cell_sha256(authority_row),
            }
            for identity, authority_row in zip(compact_identities, authority_cells)
        ],
        "chunks": [{
            "chunk_id": "chunk_3",
            "cells": compact_identities,
            "membership_sha256": dm.chunk_membership_sha256(compact_identities),
        }],
    }
    matrix["matrix_seal_sha256"] = dm._canonical_sha256(matrix)
    _write_json(matrix_path, matrix)
    budget_path = inputs / "budget.json"
    _write_json(budget_path, {"retention_budget_gib": 200})
    preflight_path = inputs / "preflight.json"
    ncu_path = inputs / "ncu"
    ncu_path.write_bytes(b"fixture-ncu")
    ncu_path.chmod(0o755)
    ncu_identity = {
        "resolved_path": str(ncu_path),
        "executable_sha256": _sha(ncu_path),
        "version_line": "Version 2026.1.0.0 (public-release)",
    }
    preflight_generated_at = "2026-07-10T12:00:00Z"
    admitted_at = "2026-07-10T12:00:01Z"
    capture_started_at = "2026-07-10T12:00:02Z"
    _write_json(preflight_path, {
        "arch": "sm_86",
        "git_head": head,
        "generated_at_utc": preflight_generated_at,
        "working_tree_clean": True,
        "pins_sha256": _sha(pins_path),
        "capture_admitted": True,
        "blockers": [],
        "checks": {"ncu": ncu_identity},
    })

    output_root = tmp_path / "capture"
    results: list[dict] = []
    membership: list[dict] = []
    for (source, slug), authority_row in zip(identities, authority_cells):
        authority_sha = dm.authority_cell_sha256(authority_row)
        membership.append({
            "roster_source": source,
            "slug": slug,
            "authority_cell_sha256": authority_sha,
        })
        artifacts: list[dict] = []
        for kind, filename in (
            ("label_sidecar", "labels.json"),
            ("label_run_manifest", "label_run_manifest.json"),
            ("ncu_report", "profile.ncu-rep"),
            ("ncu_launch_observation", "ncu_launch_observation.json"),
            ("execution_log", "label_capture_manifest.json"),
            ("trace_manifest", "trace_manifest.json"),
            ("trace_root", "dynamic_trace.pb.zst"),
            ("trace_stats", "stats.csv"),
            ("trace_threadblock", "threadblocks/0.pb"),
            ("control_bits", "control_bits.json"),
            ("control_manifest", "control_manifest.json"),
            ("cell_result", "cell_result.json"),
        ):
            rel = f"{source}/{slug}/{filename}"
            path = output_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if kind == "cell_result":
                _write_json(path, {
                    "schema_version": 1,
                    "identity": {"roster_source": source, "slug": slug},
                    "authority_cell_sha256": authority_sha,
                    "disposition": "admitted",
                    "typed_reason": None,
                    "capture_preflight_sha256": _sha(preflight_path),
                    "capture_started_at_utc": capture_started_at,
                    "admission_run_id": "run-1",
                    "supersedes": None,
                    "prior_preparation_failures": [],
                    "prior_capture_defect_attempts": [],
                    "definition_equivalence": None,
                    "definition_equivalence_attestation_sha256": None,
                })
            elif kind == "ncu_launch_observation":
                _write_json(path, {
                    "report_sha256": _sha(path.parent / "profile.ncu-rep"),
                    "observation_sha256": dm._canonical_sha256({
                        "report_sha256": _sha(path.parent / "profile.ncu-rep"),
                    }),
                })
            elif kind == "execution_log":
                _write_json(path, {
                    "schema_version": 1,
                    "command": ["label-driver", "--", "fixture-binary"],
                    "collection_environment": {
                        "LANG": "C", "LANGUAGE": "C", "LC_ALL": "C",
                        "PYTHONNOUSERSITE": "1",
                        "LABEL_DRIVER_GPU_INDEX": "0",
                        "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
                        "PATH": "/usr/local/cuda-13.2/bin:/usr/bin:/bin",
                        "PWD": str(repo),
                    },
                    "working_directory": str(repo),
                })
            else:
                if kind == "label_run_manifest":
                    _write_json(path, {
                        "schema_version": 1,
                        "ncu_identity": ncu_identity,
                    })
                else:
                    path.write_bytes(f"{source}:{slug}:{kind}\n".encode())
            artifacts.append({
                "artifact_kind": kind,
                "relative_path": rel,
                "sha256": _sha(path),
                "size_bytes": path.stat().st_size,
            })
        artifact_by_kind = {item["artifact_kind"]: item for item in artifacts}
        receipt_path = output_root / source / slug / "logs" / "label_ncu_collection_receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt = ncu_collection_receipt.build_receipt(
            preflight_sha256=_sha(preflight_path),
            expected_ncu_identity=ncu_action_executor.build_expected_ncu_identity(
                **ncu_identity,
            ),
            command=["label-driver", "--", "fixture-binary"],
            environment=ncu_action_executor.NcuCollectionEnvironment({
                "LANG": "C", "LANGUAGE": "C", "LC_ALL": "C",
                "PYTHONNOUSERSITE": "1", "LABEL_DRIVER_GPU_INDEX": "0",
                "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
                "PATH": "/usr/local/cuda-13.2/bin:/usr/bin:/bin",
                "PWD": str(repo),
            }),
            working_directory=repo,
            exit_status=0,
            artifacts=[
                ncu_collection_receipt.artifact_binding(
                    role=role,
                    relative_path=artifact_by_kind[kind]["relative_path"].split(
                        f"{source}/{slug}/", 1
                    )[1],
                    sha256=artifact_by_kind[kind]["sha256"],
                    size_bytes=artifact_by_kind[kind]["size_bytes"],
                )
                for role, kind in (
                    ("ncu_report", "ncu_report"),
                    ("label_run_manifest", "label_run_manifest"),
                    ("ncu_launch_observation", "ncu_launch_observation"),
                    ("label_capture_manifest", "execution_log"),
                )
            ],
            launch_observation_sha256=dm._canonical_sha256({
                "report_sha256": _sha(
                    output_root / source / slug / "profile.ncu-rep"
                ),
            }),
        )
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt_rel = f"{source}/{slug}/logs/label_ncu_collection_receipt.json"
        artifacts.append({
            "artifact_kind": "ncu_collection_receipt",
            "relative_path": receipt_rel,
            "sha256": _sha(receipt_path),
            "size_bytes": receipt_path.stat().st_size,
        })
        results.append({
            "roster_source": source,
            "slug": slug,
            "authority_cell_sha256": authority_sha,
            "disposition": "admitted",
            "typed_reason": None,
            "detail": None,
            "capture_preflight_sha256": _sha(preflight_path),
            "capture_started_at_utc": capture_started_at,
            "admission_run_id": "run-1",
            "supersedes": None,
            "prior_preparation_failures": [],
            "prior_capture_defect_attempts": [],
            "definition_equivalence": None,
            "definition_equivalence_attestation_sha256": None,
            "artifacts": artifacts,
        })

    input_paths = {
        "pins": str(pins_path),
        "roster_authority": str(authority_path),
        "trace_budget": str(budget_path),
        "capture_matrix": str(matrix_path),
        "preflight": str(preflight_path),
    }
    kwargs = {
        "command": "capture --chunk chunk_3",
        "source_commit_pre": head,
        "source_commit_post": head,
        "source_head_at_seal": head,
        "working_tree_clean": True,
        "tool_versions": {
            "ncu": "2026.1.0", "driver": "610.43.02",
            "nvcc": "13.3", "nvbit": "1.8",
        },
        "output_root": str(output_root),
        "admitted": 2,
        "excluded": 0,
        "expected_cells": 2,
        "chunk_membership": membership,
        "cell_results": results,
        "typed_failures": [],
        "pins_sha256": _sha(pins_path),
        "roster_authority_sha256": _sha(authority_path),
        "trace_budget_sha256": _sha(budget_path),
        "capture_matrix_sha256": _sha(matrix_path),
        "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "preflight_sha256": _sha(preflight_path),
        "admission": {
            "preflight_generated_at_utc": preflight_generated_at,
            "capture_invocations": [{
                "run_id": "run-1", "admitted_at_utc": admitted_at,
            }],
        },
        "input_paths": input_paths,
        "protocol_version": "v2.1",
        "arch": "sm_86",
        "chunk_id": "chunk_3",
        "repo_root": repo,
    }
    return {
        "repo": repo,
        "head": head,
        "inputs": inputs,
        "output_root": output_root,
        "kwargs": kwargs,
        "paths": {name: Path(path) for name, path in input_paths.items()},
    }


def _build(case: dict, **over: object) -> dict:
    kwargs = copy.deepcopy(case["kwargs"])
    kwargs.update(over)
    return dm.build_done_manifest(**kwargs)


def _cell_result_document(case: dict, manifest: dict, index: int = 0):
    row = manifest["cell_results"][index]
    artifact = next(
        item for item in row["artifacts"]
        if item["artifact_kind"] == "cell_result"
    )
    path = case["output_root"] / artifact["relative_path"]
    return row, artifact, path, json.loads(path.read_text())


def _write_and_reseal(path: Path, artifact: dict, document: dict) -> None:
    _write_json(path, document)
    artifact.update({"sha256": _sha(path), "size_bytes": path.stat().st_size})


def _append_artifact(case: dict, row: dict, *, kind: str, name: str) -> dict:
    path = case["output_root"] / row["roster_source"] / row["slug"] / name
    _write_json(path, {"kind": kind})
    artifact = {
        "artifact_kind": kind,
        "relative_path": path.relative_to(case["output_root"]).as_posix(),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }
    row["artifacts"].append(artifact)
    return artifact


def _install_supersession_evidence(
    case: dict, manifest: dict,
) -> dict:
    row, artifact, result_path, result = _cell_result_document(case, manifest)
    original_preflight = "a" * 64
    superseded_root = (
        case["output_root"].parent / ".staging" / "head" / "chunk" /
        "matrix" / "superseded"
    )
    destination = (
        superseded_root /
        (
            "20260710T120500Z__pf-"
            f"{original_preflight[:8]}__to-pf-{manifest['preflight_sha256'][:8]}"
        ) /
        f"{row['roster_source']}__{row['slug']}"
    )
    destination.mkdir(parents=True)
    moved = destination / "journal_start.json"
    journal_start = {
        "schema_version": 2,
        "state": "capture_started",
        "identity": {
            "roster_source": row["roster_source"], "slug": row["slug"],
        },
        "authority_cell_sha256": row["authority_cell_sha256"],
        "source_commit": manifest["source_commit_pre"],
        "capture_matrix_sha256": manifest["capture_matrix_sha256"],
        "matrix_seal_sha256": manifest["matrix_seal_sha256"],
        "capture_preflight_sha256": original_preflight,
        "capture_preflight_path": str(
            (case["inputs"] / "expired-preflight.json").resolve()
        ),
        "capture_started_at_utc": "2026-07-10T12:01:00Z",
        "admission_run_id": "expired-run",
        "supersedes": None,
    }
    journal_start["journal_start_seal_sha256"] = dm._canonical_sha256(
        journal_start
    )
    _write_json(moved, journal_start)
    inventory = {
        "schema_version": 1,
        "files": [{
            "relative_path": "journal_start.json",
            "sha256": _sha(moved),
            "size_bytes": moved.stat().st_size,
        }],
    }
    inventory_path = destination / "inventory.json"
    _write_json(inventory_path, inventory)
    entry = {
        "schema_version": 1,
        "timestamp_utc": "2026-07-10T12:05:00Z",
        "session_id": "session-1",
        "chunk_id": manifest["chunk_id"],
        "cell_id": f"{row['roster_source']}__{row['slug']}",
        "original_preflight_sha256": original_preflight,
        "current_preflight_sha256": manifest["preflight_sha256"],
        "source_path": str(
            (superseded_root.parent / "journals" /
             f"{row['roster_source']}__{row['slug']}").resolve()
        ),
        "destination_path": str(destination.resolve()),
        "reason_code": dm._SUPERSESSION_REASON,
        "operator_flag": "--supersede-stale-journals",
        "source_commit": manifest["source_commit_pre"],
        "matrix_seal_sha256": manifest["matrix_seal_sha256"],
        "inventory_sha256": _sha(inventory_path),
    }
    ledger_entry_id = dm._canonical_sha256(entry)
    entry["ledger_entry_id"] = ledger_entry_id
    backref = {
        "ledger_entry_id": ledger_entry_id,
        "superseded_preflight_sha256": original_preflight,
        "superseded_path": str(destination.resolve()),
    }
    row["supersedes"] = backref
    result["supersedes"] = backref
    _write_and_reseal(result_path, artifact, result)
    ledger_path = superseded_root / "supersession_ledger.jsonl"
    ledger_path.write_text(json.dumps(entry, sort_keys=True) + "\n")
    return {
        "row": row,
        "result": result,
        "result_artifact": artifact,
        "result_path": result_path,
        "destination": destination,
        "inventory_path": inventory_path,
        "ledger_path": ledger_path,
        "entry": entry,
    }


def _upgrade_case_to_current_sol(case: dict) -> None:
    """Turn cell_b into a current-matrix SOL row with full sealed attestation."""
    repo_fingerprint = case["repo"] / dm.SOL_NATIVE_ENV_PIN_PATH
    repo_fingerprint.parent.mkdir(parents=True, exist_ok=True)
    repo_fingerprint.write_bytes((_REPO / dm.SOL_NATIVE_ENV_PIN_PATH).read_bytes())
    _git(case["repo"], "add", dm.SOL_NATIVE_ENV_PIN_PATH)
    _git(case["repo"], "commit", "-qm", "pin SOL fingerprint")
    head = _git(case["repo"], "rev-parse", "HEAD")
    case["head"] = head
    for key in ("source_commit_pre", "source_commit_post", "source_head_at_seal"):
        case["kwargs"][key] = head

    authority_path = case["paths"]["roster_authority"]
    authority = json.loads(authority_path.read_text())
    authority["source_commit"] = head
    authority["cells"][1]["roster_source"] = "sol_execbench"
    authority_sha = dm.authority_cell_sha256(authority["cells"][1])
    _write_json(authority_path, authority)

    membership = case["kwargs"]["chunk_membership"][1]
    membership["roster_source"] = "sol_execbench"
    membership["authority_cell_sha256"] = authority_sha
    result = case["kwargs"]["cell_results"][1]
    result["roster_source"] = "sol_execbench"
    result["authority_cell_sha256"] = authority_sha
    old_dir = case["output_root"] / "simbench_bw" / "cell_b"
    new_dir = case["output_root"] / "sol_execbench" / "cell_b"
    new_dir.parent.mkdir(parents=True, exist_ok=True)
    old_dir.rename(new_dir)
    for artifact in result["artifacts"]:
        artifact["relative_path"] = artifact["relative_path"].replace(
            "simbench_bw/cell_b/", "sol_execbench/cell_b/", 1,
        )

    pin = {
        "path": dm.SOL_NATIVE_ENV_PIN_PATH,
        "sha256": _sha(repo_fingerprint),
        "native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
    }
    attestation_path = new_dir / dm.SOL_NATIVE_ENV_ATTESTATION_FILENAME
    _write_json(
        attestation_path,
        _sol_attestation_document(
            {"roster_source": "sol_execbench", "slug": "cell_b"}, pin,
        ),
    )
    result["artifacts"].append({
        "artifact_kind": "sol_capture_evidence",
        "relative_path": attestation_path.relative_to(
            case["output_root"]
        ).as_posix(),
        "sha256": _sha(attestation_path),
        "size_bytes": attestation_path.stat().st_size,
    })

    stage = new_dir / "sol_stage"
    stage.mkdir()
    source_definition = b'{"name":"done-v2-fixture"}\n'
    staged_definition = bytes(source_definition)
    source_path = stage / dm.SOL_DEFINITION_SOURCE_FILENAME
    staged_path = stage / dm.SOL_DEFINITION_STAGED_FILENAME
    source_path.write_bytes(source_definition)
    staged_path.write_bytes(staged_definition)
    definition_attestation = {
        "rule_id": dm.SOL_DEFINITION_EQUIVALENCE_RULE_ID,
        "result": "equal",
        "source_definition_sha256": _sha(source_path),
        "staged_definition_sha256": _sha(staged_path),
        "top_level_field_states": {
            field: {
                "source": "missing", "staged": "missing",
                "source_normalized": True, "staged_normalized": True,
            }
            for field in sorted(dm._SOL_DEFINITION_TOP_LEVEL_FIELDS)
        },
        "axes_container_state": {"source": "missing", "staged": "missing"},
        "axis_description_states": {},
        "tensor_container_states": {
            name: {"source": "missing", "staged": "missing"}
            for name in ("inputs", "outputs")
        },
        "tensor_description_states": {},
    }
    definition_attestation_path = stage / dm.SOL_DEFINITION_EQUIVALENCE_FILENAME
    definition_attestation_path.write_bytes(
        dm._canonical_definition_attestation_bytes(definition_attestation)
    )
    definition_attestation_sha = _sha(definition_attestation_path)
    prepare_manifest = {
        "source_definition_sha256": _sha(source_path),
        "staged_definition_sha256": _sha(staged_path),
        "definition_equivalence": definition_attestation,
        "artifacts": {
            "definition": {
                "path": dm.SOL_DEFINITION_STAGED_FILENAME,
                "sha256": _sha(staged_path),
            },
            "definition_source": {
                "path": dm.SOL_DEFINITION_SOURCE_FILENAME,
                "sha256": _sha(source_path),
            },
            "definition_equivalence_attestation": {
                "path": dm.SOL_DEFINITION_EQUIVALENCE_FILENAME,
                "sha256": definition_attestation_sha,
            },
        },
    }
    prepare_manifest["stage_definition_sha256"] = dm._sol_stage_manifest_sha256(
        prepare_manifest
    )
    prepare_manifest_path = stage / dm.SOL_DEFINITION_PREPARE_MANIFEST_FILENAME
    _write_json(prepare_manifest_path, prepare_manifest)
    for path in (
        source_path, staged_path, definition_attestation_path,
        prepare_manifest_path,
    ):
        result["artifacts"].append({
            "artifact_kind": "sol_capture_evidence",
            "relative_path": path.relative_to(case["output_root"]).as_posix(),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        })
    result["definition_equivalence"] = definition_attestation
    result["definition_equivalence_attestation_sha256"] = (
        definition_attestation_sha
    )

    per_cell_path = case["inputs"] / "per_cell.json"
    registry_path = case["inputs"] / "registry.json"
    _write_json(per_cell_path, {"schema_version": 1, "cells": []})
    _write_json(registry_path, {"schema_version": 1, "metrics": []})
    matrix_path = case["paths"]["capture_matrix"]
    matrix = json.loads(matrix_path.read_text())
    matrix["source_commit"] = head
    matrix["cells"][1].update({
        "identity": {"roster_source": "sol_execbench", "slug": "cell_b"},
        "authority_cell_sha256": authority_sha,
        "invocation": {
            "adapter_contract": {
                "native_env_hash": dm.SOL_CERTIFIED_NATIVE_ENV_HASH,
                "environment": dict(dm.SOL_CERTIFIED_ENVIRONMENT),
            },
            "logical_workload": {
                "definition_file_sha256": _sha(source_path),
            },
        },
    })
    matrix["chunks"][0]["cells"][1]["roster_source"] = "sol_execbench"
    matrix["chunks"][0]["membership_sha256"] = dm.chunk_membership_sha256(
        matrix["chunks"][0]["cells"]
    )
    matrix["input_pins"] = {
        "protocol_pins": {
            "path": str(case["paths"]["pins"]),
            "sha256": _sha(case["paths"]["pins"]),
        },
        "aggregate_retention_budget": {
            "path": str(case["paths"]["trace_budget"]),
            "sha256": _sha(case["paths"]["trace_budget"]),
        },
        "per_cell_trace_budgets": {
            "path": str(per_cell_path), "sha256": _sha(per_cell_path),
        },
        "metric_registry": {
            "path": str(registry_path), "sha256": _sha(registry_path),
        },
        dm.SOL_NATIVE_ENV_PIN_SECTION: pin,
    }
    matrix.pop("matrix_seal_sha256", None)
    matrix["matrix_seal_sha256"] = dm._canonical_sha256(matrix)
    _write_json(matrix_path, matrix)

    preflight_path = case["paths"]["preflight"]
    preflight = json.loads(preflight_path.read_text())
    preflight["git_head"] = head
    _write_json(preflight_path, preflight)
    preflight_sha = _sha(preflight_path)
    for result in case["kwargs"]["cell_results"]:
        receipt_artifact = next(
            artifact for artifact in result["artifacts"]
            if artifact["artifact_kind"] == "ncu_collection_receipt"
        )
        receipt_path = case["output_root"] / receipt_artifact["relative_path"]
        receipt = json.loads(receipt_path.read_text())
        receipt.pop("receipt_seal_sha256", None)
        receipt["preflight_sha256"] = preflight_sha
        receipt = dm.immutable.seal_model(receipt, "receipt_seal_sha256")
        _write_json(receipt_path, receipt)
        receipt_artifact.update({
            "sha256": _sha(receipt_path), "size_bytes": receipt_path.stat().st_size,
        })
    for result in case["kwargs"]["cell_results"]:
        result["capture_preflight_sha256"] = preflight_sha
        cell_artifact = next(
            artifact for artifact in result["artifacts"]
            if artifact["artifact_kind"] == "cell_result"
        )
        cell_path = case["output_root"] / cell_artifact["relative_path"]
        cell_document = json.loads(cell_path.read_text())
        cell_document.update({
            "identity": {
                "roster_source": result["roster_source"],
                "slug": result["slug"],
            },
            "authority_cell_sha256": result["authority_cell_sha256"],
            "capture_preflight_sha256": preflight_sha,
            "definition_equivalence": result["definition_equivalence"],
            "definition_equivalence_attestation_sha256": result[
                "definition_equivalence_attestation_sha256"
            ],
        })
        _write_json(cell_path, cell_document)
        cell_artifact.update({
            "sha256": _sha(cell_path), "size_bytes": cell_path.stat().st_size,
        })
    case["kwargs"].update({
        "roster_authority_sha256": _sha(authority_path),
        "capture_matrix_sha256": _sha(matrix_path),
        "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "preflight_sha256": preflight_sha,
    })


def test_valid_manifest_is_v2_schema_valid_and_resume_safe(case):
    manifest = _build(case)
    jsonschema.validate(manifest, _SCHEMA)
    assert manifest["schema_version"] == 2
    assert manifest["counts"] == {"admitted": 2, "excluded": 0, "failed": 0}
    assert dm.verify_seal(manifest, repo_root=case["repo"]) == []


def test_done_rejects_missing_ncu_collection_receipt(case):
    case["kwargs"]["cell_results"][0]["artifacts"] = [
        artifact for artifact in case["kwargs"]["cell_results"][0]["artifacts"]
        if artifact["artifact_kind"] != "ncu_collection_receipt"
    ]
    with pytest.raises(dm.DoneManifestError, match="ncu_collection_receipt"):
        _build(case)


def test_done_rejects_resealed_ncu_identity_drift(case):
    manifest = _build(case)
    row = manifest["cell_results"][0]
    artifact = next(
        item for item in row["artifacts"]
        if item["artifact_kind"] == "ncu_collection_receipt"
    )
    path = case["output_root"] / artifact["relative_path"]
    receipt = json.loads(path.read_text())
    receipt.pop("receipt_seal_sha256", None)
    receipt["ncu_identity"]["executable_sha256"] = "f" * 64
    _write_and_reseal(path, artifact, dm.immutable.seal_model(
        receipt, "receipt_seal_sha256"
    ))
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("receipt identity differs" in problem for problem in problems)


def test_done_rejects_resealed_ncu_path_substitution(case):
    manifest = _build(case)
    row = manifest["cell_results"][0]
    artifact = next(
        item for item in row["artifacts"]
        if item["artifact_kind"] == "ncu_collection_receipt"
    )
    path = case["output_root"] / artifact["relative_path"]
    receipt = json.loads(path.read_text())
    receipt.pop("receipt_seal_sha256", None)
    receipt["ncu_identity"]["resolved_path"] = "/substituted/ncu"
    _write_and_reseal(path, artifact, dm.immutable.seal_model(
        receipt, "receipt_seal_sha256"
    ))
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("receipt identity differs" in problem for problem in problems)


def test_done_rejects_resealed_ncu_artifact_hash_drift(case):
    manifest = _build(case)
    row = manifest["cell_results"][0]
    artifact = next(
        item for item in row["artifacts"]
        if item["artifact_kind"] == "ncu_collection_receipt"
    )
    path = case["output_root"] / artifact["relative_path"]
    receipt = json.loads(path.read_text())
    receipt.pop("receipt_seal_sha256", None)
    report_binding = next(
        item for item in receipt["artifacts"]
        if item["role"] == "ncu_report"
    )
    report_binding["sha256"] = "e" * 64
    _write_and_reseal(path, artifact, dm.immutable.seal_model(
        receipt, "receipt_seal_sha256"
    ))
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("receipt ncu_report hash differs" in problem for problem in problems)


def test_done_rejects_missing_run_manifest_ncu_identity(case):
    manifest = _build(case)
    row = manifest["cell_results"][0]
    artifact = next(
        item for item in row["artifacts"]
        if item["artifact_kind"] == "label_run_manifest"
    )
    path = case["output_root"] / artifact["relative_path"]
    run = json.loads(path.read_text())
    run.pop("ncu_identity", None)
    _write_and_reseal(path, artifact, run)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("run manifest NCU identity differs" in problem for problem in problems)


def test_done_rejects_malformed_preflight_checks(case):
    manifest = _build(case)
    row = manifest["cell_results"][0]
    problems = dm._verify_ncu_collection_receipt(
        row,
        manifest=manifest,
        preflight={"checks": "malformed"},
        output_root=case["output_root"],
        label="cell_results[0]",
    )
    assert problems == ["cell_results[0] preflight lacks checks.ncu authority"]


def test_round28_schema_requires_admission_and_matrix_seal(case):
    manifest = _build(case)
    manifest.pop("admission")
    manifest.pop("matrix_seal_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, _SCHEMA)


def test_builder_refuses_vacuous_admission_invocation_set(case):
    admission = copy.deepcopy(case["kwargs"]["admission"])
    admission["capture_invocations"] = []
    with pytest.raises(dm.DoneManifestError, match="nonempty array"):
        _build(case, admission=admission)


@pytest.mark.parametrize(
    ("generated_at", "admitted_at", "message"),
    [
        ("2026-07-10T12:00:00Z", "2026-07-10T12:15:00.001Z", "window"),
        ("2026-07-10T12:00:03Z", "2026-07-10T12:00:00Z", "future-skew"),
    ],
)
def test_done_admission_rederives_freshness_and_clock_skew(
    case, generated_at: str, admitted_at: str, message: str,
):
    admission = copy.deepcopy(case["kwargs"]["admission"])
    admission["preflight_generated_at_utc"] = generated_at
    admission["capture_invocations"][0]["admitted_at_utc"] = admitted_at
    with pytest.raises(dm.DoneManifestError, match=message):
        _build(case, admission=admission)


def test_admission_preflight_generation_must_match_sealed_preflight(case):
    manifest = _build(case)
    manifest["admission"]["preflight_generated_at_utc"] = (
        "2026-07-10T11:59:59Z"
    )
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("does not match the sealed preflight" in problem for problem in problems)


@pytest.mark.parametrize(
    ("outcome", "route_method", "failed"),
    [
        ("ordinary", None, False),
        ("routed_ac9", "lossless_compressed_full", False),
        ("reduced_typed", "reduced_shape_matched_label", False),
        ("budget_limited_wide_ci", "unavailable", True),
        ("recovered", None, False),
    ],
)
def test_every_terminal_outcome_rejects_cell_preflight_mismatch(
    case, monkeypatch, outcome: str, route_method: str | None, failed: bool,
):
    manifest = _build(case)
    row, artifact, path, result = _cell_result_document(case, manifest)
    result["outcome_test_class"] = outcome
    if route_method is not None:
        result.update({
            "trace_route_outcome": "typed_unavailable" if failed else "admitted",
            "trace_route_method": route_method,
            "trace_route_result_sha256": "9" * 64,
        })
        _append_artifact(
            case, row, kind="trace_route_record", name=f"{outcome}_route.json",
        )
        monkeypatch.setattr(dm, "_verify_trace_route_evidence", lambda *a, **k: [])
    if failed:
        row.update({"disposition": "failed", "typed_reason": "trace_unsupported"})
        result.update({"disposition": "failed", "typed_reason": "trace_unsupported"})
        _append_artifact(
            case, row, kind="trace_auxiliary", name="budget_limited_evidence.json",
        )
        manifest["counts"] = {"admitted": 1, "excluded": 0, "failed": 1}
        manifest["typed_failures"] = [{
            "roster_source": row["roster_source"],
            "slug": row["slug"],
            "typed_reason": "trace_unsupported",
            "detail": row["detail"],
        }]
        monkeypatch.setattr(
            dm, "_verify_trace_unsupported_evidence", lambda *a, **k: [],
        )
    if outcome == "recovered":
        result["recovered_from_journal"] = True
    result["capture_preflight_sha256"] = "f" * 64
    _write_and_reseal(path, artifact, result)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any(
        "cell result provenance mismatch" in problem
        and "capture_preflight_sha256" in problem
        for problem in problems
    )


def test_cell_capture_timestamp_and_run_are_bound_to_admission(case):
    manifest = _build(case)
    _, artifact, path, result = _cell_result_document(case, manifest)
    result["capture_started_at_utc"] = "2026-07-10T12:00:00Z"
    result["admission_run_id"] = "not-admitted"
    _write_and_reseal(path, artifact, result)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("absent from DONE admission" in problem for problem in problems)
    result["admission_run_id"] = "run-1"
    _write_and_reseal(path, artifact, result)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("capture started before its admission" in problem for problem in problems)


def test_supersession_backref_ledger_and_inventory_round_trip(case):
    manifest = _build(case)
    _install_supersession_evidence(case, manifest)
    assert dm.verify_seal(manifest, repo_root=case["repo"]) == []


@pytest.mark.parametrize("damage", ["missing_ledger", "inventory_hash", "dangling_backref"])
def test_supersession_ledger_inconsistency_fails_closed(case, damage: str):
    manifest = _build(case)
    evidence = _install_supersession_evidence(case, manifest)
    if damage == "missing_ledger":
        evidence["ledger_path"].unlink()
    elif damage == "inventory_hash":
        entry = dict(evidence["entry"])
        entry["inventory_sha256"] = "e" * 64
        evidence["ledger_path"].write_text(json.dumps(entry, sort_keys=True) + "\n")
    else:
        result = dict(evidence["result"])
        result["supersedes"] = dict(result["supersedes"])
        result["supersedes"]["ledger_entry_id"] = "2" * 64
        _write_and_reseal(
            evidence["result_path"], evidence["result_artifact"], result,
        )
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("supersession" in problem.lower() for problem in problems)


def test_extra_unreferenced_ledger_row_is_rejected(case):
    manifest = _build(case)
    evidence = _install_supersession_evidence(case, manifest)
    extra = {
        key: value for key, value in evidence["entry"].items()
        if key != "ledger_entry_id"
    }
    extra["timestamp_utc"] = "2026-07-10T12:05:01Z"
    extra["ledger_entry_id"] = dm._canonical_sha256(extra)
    with evidence["ledger_path"].open("a") as stream:
        stream.write(json.dumps(extra, sort_keys=True) + "\n")
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any(
        "duplicate current ledger authority" in problem
        or "non-unique current-attempt authority" in problem
        for problem in problems
    )


def test_supersession_ledger_source_path_binds_original_journal(case):
    manifest = _build(case)
    evidence = _install_supersession_evidence(case, manifest)
    entry = {
        key: value for key, value in evidence["entry"].items()
        if key != "ledger_entry_id"
    }
    entry["source_path"] = str(
        (evidence["destination"].parent.parent.parent / "journals" / "wrong-cell")
        .resolve()
    )
    entry_id = dm._canonical_sha256(entry)
    entry["ledger_entry_id"] = entry_id
    evidence["ledger_path"].write_text(json.dumps(entry, sort_keys=True) + "\n")
    backref = dict(evidence["row"]["supersedes"])
    backref["ledger_entry_id"] = entry_id
    evidence["row"]["supersedes"] = backref
    result = dict(evidence["result"])
    result["supersedes"] = backref
    _write_and_reseal(
        evidence["result_path"], evidence["result_artifact"], result,
    )
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("source_path=" in problem and "expected=" in problem for problem in problems)


def test_unledgered_superseded_attempt_directory_is_rejected(case):
    manifest = _build(case)
    evidence = _install_supersession_evidence(case, manifest)
    sibling = evidence["destination"].parent / "unledgered-cell"
    shutil.copytree(evidence["destination"], sibling)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("ledger/destination set mismatch" in problem for problem in problems)


def test_production_stage_superseded_evidence_is_discovered_without_backref(case):
    manifest = _build(case)
    campaign_root = case["output_root"].parent / "campaign"
    final_root = campaign_root / "chunks" / manifest["chunk_id"]
    final_root.parent.mkdir(parents=True)
    case["output_root"].rename(final_root)
    manifest["output_root"] = str(final_root)
    unledgered = (
        campaign_root / ".staging" / manifest["source_commit_pre"] /
        manifest["chunk_id"] / manifest["capture_matrix_sha256"] /
        "superseded" / "20260710T120500Z__pf-aaaaaaaa__to-pf-bbbbbbbb" /
        "portable_lcd__cell_a"
    )
    unledgered.mkdir(parents=True)
    _write_json(unledgered / "journal_start.json", {"forensic": True})
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("superseded evidence has no ledger" in problem for problem in problems)


def test_published_artifact_cannot_resolve_under_superseded(case):
    manifest = _build(case)
    manifest["cell_results"][0]["artifacts"][0]["relative_path"] = (
        "superseded/attempt/cell/labels.json"
    )
    problems = dm.verify_seal(
        manifest, repo_root=case["repo"], rehash_references=False,
        require_current_head=False, require_clean_tree=False,
    )
    assert any("publishes an artifact beneath superseded" in problem for problem in problems)


def test_output_root_symlink_cannot_hide_superseded_publication(case):
    manifest = _build(case)
    original = case["output_root"]
    destination = original.parent / "superseded" / "forged-publication"
    destination.parent.mkdir()
    original.rename(destination)
    original.symlink_to(destination, target_is_directory=True)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("output_root is a symlink" in problem for problem in problems)
    assert any("resolves beneath superseded" in problem for problem in problems)


def test_output_root_symlink_loop_returns_refusal_not_exception(case):
    manifest = _build(case)
    original = case["output_root"]
    original.rename(original.with_name("preserved-capture"))
    original.symlink_to(original, target_is_directory=True)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("cannot resolve published output_root" in problem for problem in problems)


def test_superseded_path_symlink_loop_returns_typed_problem(case):
    manifest = _build(case)
    row, artifact, result_path, result = _cell_result_document(case, manifest)
    destination = (
        case["output_root"].parent / "work" / "superseded" /
        ("20260710T120500Z__pf-aaaaaaaa__to-pf-"
         f"{manifest['preflight_sha256'][:8]}") /
        f"{row['roster_source']}__{row['slug']}"
    )
    destination.parent.mkdir(parents=True)
    destination.symlink_to(destination, target_is_directory=True)
    backref = {
        "ledger_entry_id": "1" * 64,
        "superseded_preflight_sha256": "a" * 64,
        "superseded_path": str(destination),
    }
    row["supersedes"] = backref
    result["supersedes"] = backref
    _write_and_reseal(result_path, artifact, result)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("SUPERSESSION_LEDGER_INCONSISTENT" in problem for problem in problems)


@pytest.mark.campaign_evidence
def test_only_exact_historical_bootstrap_receives_schema_compatibility():
    path = Path(
        "/home/jakeshea/sm86_reference_capture_r27/chunks/"
        "portable_lcd-000-bootstrap/DONE.json"
    )
    manifest = json.loads(path.read_text())
    assert dm._is_historical_bootstrap(manifest)
    assert dm.verify_seal(
        manifest, repo_root=_REPO,
        require_current_head=False, require_clean_tree=False,
    ) == []
    tampered = copy.deepcopy(manifest)
    tampered["generated_at_utc"] = "2026-07-10T09:52:17Z"
    assert not dm._is_historical_bootstrap(tampered)
    assert any("admission" in problem for problem in dm._schema_problems(tampered))


@pytest.mark.campaign_evidence
def test_historical_bootstrap_strictly_loads_global_preparation_root(
    monkeypatch: pytest.MonkeyPatch,
):
    import archive_preparation_failure as archive

    path = Path(
        "/home/jakeshea/sm86_reference_capture_r27/chunks/"
        "portable_lcd-000-bootstrap/DONE.json"
    )
    manifest = json.loads(path.read_text())
    assert dm._is_historical_bootstrap(manifest)
    expected_ledger = (
        Path(manifest["output_root"]).parent.parent
        / dm.PREPARATION_FAILURE_ROOT_NAME
        / dm.PREPARATION_FAILURE_LEDGER_FILENAME
    )
    unledgered = expected_ledger.parent / "post-move-unledgered-attempt"
    calls: list[tuple[Path, bool]] = []

    def refuse_fs_pre(
        ledger_path: Path, *, verify_archives: bool = False,
    ) -> list[dict]:
        calls.append((ledger_path, verify_archives))
        raise archive.PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT",
            f"archive evidence exists without ledger: {[str(unledgered)]!r}",
        )

    monkeypatch.setattr(
        archive, "load_preparation_failure_ledger", refuse_fs_pre,
    )
    problems = dm.verify_seal(
        manifest, repo_root=_REPO,
        require_current_head=False, require_clean_tree=False,
    )
    assert calls == [(expected_ledger, True)]
    assert (
        "PREPARATION_LEDGER_INCONSISTENT: archive evidence exists without "
        f"ledger: {[str(unledgered)]!r}"
    ) in problems


@pytest.mark.parametrize("disposition", ["admitted", "failed"])
def test_current_matrix_sol_env_attestation_rederives_exact_contract(
    tmp_path: Path, disposition: str,
):
    attested = _sol_env_attestation_case(tmp_path, disposition=disposition)
    assert dm.verify_manifest_sol_native_env_attestations(
        attested["manifest"], attested["matrix"], repo_root=_REPO,
    ) == []


def test_schema_requires_named_sol_env_evidence_for_terminal_sol(case):
    manifest = _build(case)
    manifest["cell_results"][0]["roster_source"] = "sol_execbench"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, _SCHEMA)


def test_current_matrix_sol_env_attestation_missing_fails_closed(tmp_path: Path):
    attested = _sol_env_attestation_case(tmp_path)
    attested["manifest"]["cell_results"][0]["artifacts"] = []
    problems = dm.verify_manifest_sol_native_env_attestations(
        attested["manifest"], attested["matrix"], repo_root=_REPO,
    )
    assert any("requires exactly one sealed" in problem for problem in problems)


def test_current_matrix_sol_env_attestation_tamper_fails_closed(tmp_path: Path):
    attested = _sol_env_attestation_case(tmp_path)
    document = json.loads(attested["attestation_path"].read_text())
    document["compiler_identity"]["version_line"] = "tampered compiler"
    _write_json(attested["attestation_path"], document)
    problems = dm.verify_manifest_sol_native_env_attestations(
        attested["manifest"], attested["matrix"], repo_root=_REPO,
    )
    assert any("artifact hash is stale" in problem for problem in problems)
    assert any("attestation seal is stale" in problem for problem in problems)


def test_self_consistent_sol_env_reseal_cannot_replace_certified_identity(
    tmp_path: Path,
):
    attested = _sol_env_attestation_case(tmp_path)
    document = json.loads(attested["attestation_path"].read_text())
    document["components"]["torch_version"] = "replacement-environment"
    replacement_hash = dm._canonical_sha256(document["components"])
    document["components_sha256"] = replacement_hash
    document["observed_native_env_hash"] = replacement_hash
    document["certified_native_env_hash"] = replacement_hash
    document["matrix_pin"]["native_env_hash"] = replacement_hash
    unsigned = {
        key: value for key, value in document.items()
        if key != "attestation_sha256"
    }
    document["attestation_sha256"] = dm._canonical_sha256(unsigned)
    _write_json(attested["attestation_path"], document)
    attested["artifact"]["sha256"] = _sha(attested["attestation_path"])
    attested["artifact"]["size_bytes"] = attested["attestation_path"].stat().st_size
    problems = dm.verify_manifest_sol_native_env_attestations(
        attested["manifest"], attested["matrix"], repo_root=_REPO,
    )
    assert any("certified_native_env_hash" in problem for problem in problems)
    assert any("components rehash" in problem for problem in problems)
    assert any("matrix_pin" in problem for problem in problems)


def test_sol_env_attestation_requires_canonical_compiler_block(tmp_path: Path):
    attested = _sol_env_attestation_case(tmp_path)
    document = json.loads(attested["attestation_path"].read_text())
    document["compiler_identity"].pop("executable_sha256")
    unsigned = {
        key: value for key, value in document.items()
        if key != "attestation_sha256"
    }
    document["attestation_sha256"] = dm._canonical_sha256(unsigned)
    _write_json(attested["attestation_path"], document)
    attested["artifact"]["sha256"] = _sha(attested["attestation_path"])
    attested["artifact"]["size_bytes"] = attested["attestation_path"].stat().st_size
    problems = dm.verify_manifest_sol_native_env_attestations(
        attested["manifest"], attested["matrix"], repo_root=_REPO,
    )
    assert any("canonical compiler identity" in problem for problem in problems)


def test_legacy_matrix_does_not_retrofit_attestation_requirement(tmp_path: Path):
    attested = _sol_env_attestation_case(tmp_path)
    legacy = copy.deepcopy(attested["matrix"])
    legacy.pop("input_pins")
    attested["manifest"]["cell_results"][0]["artifacts"] = []
    assert dm.verify_manifest_sol_native_env_attestations(
        attested["manifest"], legacy, repo_root=_REPO,
    ) == []


def test_full_done_verifier_revalidates_current_sol_attestation(case):
    _upgrade_case_to_current_sol(case)
    manifest = _build(case)
    jsonschema.validate(manifest, _SCHEMA)
    assert dm.verify_seal(manifest, repo_root=case["repo"]) == []


def test_done_rejects_v1_or_missing_sol_definition_attestation(case):
    _upgrade_case_to_current_sol(case)
    result = next(
        row for row in case["kwargs"]["cell_results"]
        if row["roster_source"] == "sol_execbench"
    )
    result["definition_equivalence"]["rule_id"] = "SOL_DEF_EQUIV_V1"
    with pytest.raises(dm.DoneManifestError, match="SOL_ATTESTATION_RULE_MISMATCH"):
        _build(case)
    result["definition_equivalence"] = None
    with pytest.raises(dm.DoneManifestError, match="SOL_ATTESTATION_RULE_MISMATCH"):
        _build(case)


def test_done_rejects_ineligible_normalized_pointer(case):
    _upgrade_case_to_current_sol(case)
    result = next(
        row for row in case["kwargs"]["cell_results"]
        if row["roster_source"] == "sol_execbench"
    )
    result["definition_equivalence"]["axis_description_states"] = {
        "bad~pointer": {
            "source": "axis_not_object", "staged": "axis_not_object",
            "source_normalized": False, "staged_normalized": False,
        },
    }
    with pytest.raises(
        dm.DoneManifestError, match="SOL_ATTESTATION_POINTER_INELIGIBLE"
    ):
        _build(case)


def test_done_rejects_malformed_tensor_normalized_pointer(case):
    _upgrade_case_to_current_sol(case)
    result = next(
        row for row in case["kwargs"]["cell_results"]
        if row["roster_source"] == "sol_execbench"
    )
    result["definition_equivalence"]["tensor_description_states"] = {
        "/inputs/bad~escape/description": {
            "source": "null",
            "staged": "null",
            "source_normalized": False,
            "staged_normalized": False,
        },
    }
    with pytest.raises(
        dm.DoneManifestError, match="SOL_ATTESTATION_POINTER_INELIGIBLE"
    ):
        _build(case)


def test_done_rejects_tampered_v2_state_after_file_reseal(case):
    _upgrade_case_to_current_sol(case)
    manifest = _build(case)
    row = next(
        item for item in manifest["cell_results"]
        if item["roster_source"] == "sol_execbench"
    )
    attestation_artifact = next(
        artifact for artifact in row["artifacts"]
        if PurePosixPath(artifact["relative_path"]).name
        == dm.SOL_DEFINITION_EQUIVALENCE_FILENAME
    )
    attestation_path = case["output_root"] / attestation_artifact["relative_path"]
    document = json.loads(attestation_path.read_text())
    document["top_level_field_states"]["op_type"]["source"] = "null"
    document["top_level_field_states"]["op_type"]["source_normalized"] = False
    raw = dm._canonical_definition_attestation_bytes(document)
    assert raw is not None
    attestation_path.write_bytes(raw)
    attestation_artifact.update({
        "sha256": _sha(attestation_path),
        "size_bytes": attestation_path.stat().st_size,
    })
    row["definition_equivalence"] = document
    row["definition_equivalence_attestation_sha256"] = _sha(attestation_path)
    problems = dm.verify_manifest_sol_definition_attestations(
        manifest, json.loads(case["paths"]["capture_matrix"].read_text()),
    )
    assert any("SOL_ATTESTATION_STATE_MISMATCH" in problem for problem in problems)


def _project_manifest_definition_attestation_to_e387(
    case: dict, manifest: dict,
) -> tuple[dict, Path]:
    row = next(
        item for item in manifest["cell_results"]
        if item["roster_source"] == "sol_execbench"
    )
    current = row["definition_equivalence"]
    projected = {
        key: copy.deepcopy(value)
        for key, value in current.items()
        if key not in {
            "tensor_container_states", "tensor_description_states",
        }
    }
    attestation_artifact = next(
        artifact for artifact in row["artifacts"]
        if PurePosixPath(artifact["relative_path"]).name
        == dm.SOL_DEFINITION_EQUIVALENCE_FILENAME
    )
    attestation_path = case["output_root"] / attestation_artifact["relative_path"]
    attestation_path.write_bytes(sol_def.serialize_e387_legacy_attestation(
        projected,
        current,
        source_commit=sol_def.E387_LEGACY_SOURCE_COMMIT,
        comparator_sha256=sol_def.E387_LEGACY_COMPARATOR_SHA256,
        capture_adapter_sha256=sol_def.E387_LEGACY_CAPTURE_ADAPTER_SHA256,
    ))
    attestation_artifact.update({
        "sha256": _sha(attestation_path),
        "size_bytes": attestation_path.stat().st_size,
    })
    row["definition_equivalence"] = projected
    row["definition_equivalence_attestation_sha256"] = _sha(attestation_path)

    prepare_artifact = next(
        artifact for artifact in row["artifacts"]
        if PurePosixPath(artifact["relative_path"]).name
        == dm.SOL_DEFINITION_PREPARE_MANIFEST_FILENAME
    )
    prepare_path = case["output_root"] / prepare_artifact["relative_path"]
    prepare = json.loads(prepare_path.read_text())
    prepare["definition_equivalence"] = projected
    prepare["artifacts"]["definition_equivalence_attestation"] = {
        "path": dm.SOL_DEFINITION_EQUIVALENCE_FILENAME,
        "sha256": _sha(attestation_path),
    }
    prepare["artifacts"]["adapter_source"] = {
        "path": (
            "/historical/checkout/"
            + dm.SOL_E387_LEGACY_CAPTURE_ADAPTER_PATH.as_posix()
        ),
        "sha256": dm.SOL_E387_LEGACY_CAPTURE_ADAPTER_SHA256,
    }
    prepare.pop("stage_definition_sha256")
    prepare["stage_definition_sha256"] = dm._sol_stage_manifest_sha256(
        prepare
    )
    _write_json(prepare_path, prepare)
    prepare_artifact.update({
        "sha256": _sha(prepare_path),
        "size_bytes": prepare_path.stat().st_size,
    })
    return projected, prepare_path


@pytest.mark.campaign_evidence
def test_exact_e387_definition_projection_verifies_with_local_git_blobs(case):
    _upgrade_case_to_current_sol(case)
    manifest = _build(case)
    projected, prepare_path = _project_manifest_definition_attestation_to_e387(
        case, manifest
    )
    manifest["source_commit_pre"] = dm.SOL_E387_LEGACY_SOURCE_COMMIT
    matrix = json.loads(case["paths"]["capture_matrix"].read_text())

    assert sol_def.is_e387_legacy_attestation_shape(projected)
    assert dm.verify_manifest_sol_definition_attestations(
        manifest,
        matrix,
        repo_root=_REPO,
        source_commit=dm.SOL_E387_LEGACY_SOURCE_COMMIT,
    ) == []

    prepare = json.loads(prepare_path.read_text())
    prepare["artifacts"]["adapter_source"]["sha256"] = "0" * 64
    prepare.pop("stage_definition_sha256")
    prepare["stage_definition_sha256"] = dm._sol_stage_manifest_sha256(
        prepare
    )
    _write_json(prepare_path, prepare)
    row = next(
        item for item in manifest["cell_results"]
        if item["roster_source"] == "sol_execbench"
    )
    prepare_artifact = next(
        artifact for artifact in row["artifacts"]
        if PurePosixPath(artifact["relative_path"]).name
        == dm.SOL_DEFINITION_PREPARE_MANIFEST_FILENAME
    )
    prepare_artifact.update({
        "sha256": _sha(prepare_path),
        "size_bytes": prepare_path.stat().st_size,
    })
    problems = dm.verify_manifest_sol_definition_attestations(
        manifest,
        matrix,
        repo_root=_REPO,
        source_commit=dm.SOL_E387_LEGACY_SOURCE_COMMIT,
    )
    assert any("exact e387 adapter" in problem for problem in problems)


def test_done_rejects_source_sha_different_from_matrix_row_pin(case):
    _upgrade_case_to_current_sol(case)
    manifest = _build(case)
    matrix = json.loads(case["paths"]["capture_matrix"].read_text())
    sol = next(
        item for item in matrix["cells"]
        if item["identity"]["roster_source"] == "sol_execbench"
    )
    sol["invocation"]["logical_workload"]["definition_file_sha256"] = "f" * 64
    problems = dm.verify_manifest_sol_definition_attestations(manifest, matrix)
    assert any(
        "source definition SHA differs from the matrix per-row pin" in problem
        for problem in problems
    )


def test_full_done_verifier_rejects_self_consistent_sol_attestation_reseal(case):
    _upgrade_case_to_current_sol(case)
    manifest = _build(case)
    result = next(
        row for row in manifest["cell_results"]
        if row["roster_source"] == "sol_execbench"
    )
    artifact = next(
        item for item in result["artifacts"]
        if PurePosixPath(item["relative_path"]).name
        == dm.SOL_NATIVE_ENV_ATTESTATION_FILENAME
    )
    path = case["output_root"] / artifact["relative_path"]
    document = json.loads(path.read_text())
    document["components"]["torch_version"] = "replacement-environment"
    replacement_hash = dm._canonical_sha256(document["components"])
    document["components_sha256"] = replacement_hash
    document["observed_native_env_hash"] = replacement_hash
    document["certified_native_env_hash"] = replacement_hash
    document["matrix_pin"]["native_env_hash"] = replacement_hash
    unsigned = {
        key: value for key, value in document.items()
        if key != "attestation_sha256"
    }
    document["attestation_sha256"] = dm._canonical_sha256(unsigned)
    _write_json(path, document)
    artifact["sha256"] = _sha(path)
    artifact["size_bytes"] = path.stat().st_size
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("certified_native_env_hash" in problem for problem in problems)
    assert any("components rehash" in problem for problem in problems)


def test_admitted_route_rederives_replay_and_all_input_bindings(tmp_path: Path):
    routed = _admitted_route_case(tmp_path)
    problems = dm._verify_trace_route_evidence(
        routed["row"], manifest=routed["manifest"], matrix=routed["matrix"],
        preflight=routed["preflight"], output_root=Path(
            routed["manifest"]["output_root"]
        ), label="cell_results[0]",
    )
    assert problems == []


def test_admitted_route_rejects_retained_payload_tamper(tmp_path: Path):
    routed = _admitted_route_case(tmp_path)
    payload = next(routed["trace_dir"].rglob("*.pb.zst"))
    payload.write_bytes(payload.read_bytes() + b"tamper")
    problems = dm._verify_trace_route_evidence(
        routed["row"], manifest=routed["manifest"], matrix=routed["matrix"],
        preflight=routed["preflight"], output_root=Path(
            routed["manifest"]["output_root"]
        ), label="cell_results[0]",
    )
    assert any("retained route inventory" in problem for problem in problems)
    assert any("replay bundle verification failed" in problem for problem in problems)


def test_self_consistent_route_reseal_cannot_replace_matrix_inputs(tmp_path: Path):
    routed = _admitted_route_case(tmp_path)
    row = routed["row"]
    by_kind = {artifact["artifact_kind"]: artifact for artifact in row["artifacts"]}
    trace_manifest_path = Path(routed["manifest"]["output_root"]) / by_kind[
        "trace_manifest"
    ]["relative_path"]
    trace_manifest = json.loads(trace_manifest_path.read_text())
    trace_manifest["binary_sha256"] = "a" * 64
    _write_json(trace_manifest_path, trace_manifest)
    by_kind["trace_manifest"]["sha256"] = _sha(trace_manifest_path)
    by_kind["trace_manifest"]["size_bytes"] = trace_manifest_path.stat().st_size

    route = json.loads(routed["route_path"].read_text())
    route["protocol"]["binary_sha256"] = "a" * 64
    route["protocol"]["capture_manifest_sha256"] = _sha(trace_manifest_path)
    route["protocol_sha256"] = dm._canonical_sha256(route["protocol"])
    route["result_sha256"] = dm._canonical_sha256({
        key: value for key, value in route.items() if key != "result_sha256"
    })
    _write_json(routed["route_path"], route)
    routed["route_artifact"]["sha256"] = _sha(routed["route_path"])
    routed["route_artifact"]["size_bytes"] = routed["route_path"].stat().st_size

    cell_result_path = Path(routed["manifest"]["output_root"]) / by_kind[
        "cell_result"
    ]["relative_path"]
    cell_result = json.loads(cell_result_path.read_text())
    cell_result["binary_sha256"] = "a" * 64
    cell_result["trace_route_result_sha256"] = route["result_sha256"]
    _write_json(cell_result_path, cell_result)
    by_kind["cell_result"]["sha256"] = _sha(cell_result_path)
    by_kind["cell_result"]["size_bytes"] = cell_result_path.stat().st_size

    problems = dm._verify_trace_route_evidence(
        row, manifest=routed["manifest"], matrix=routed["matrix"],
        preflight=routed["preflight"], output_root=Path(
            routed["manifest"]["output_root"]
        ), label="cell_results[0]",
    )
    assert any("protocol/input hashes do not rederive" in problem for problem in problems)


def test_route_outcome_method_mismatch_fails_after_valid_reseal(tmp_path: Path):
    routed = _admitted_route_case(tmp_path)
    route = json.loads(routed["route_path"].read_text())
    route["outcome"] = "typed_unavailable"
    route["typed_reason"] = "gate_forced"
    route["result_sha256"] = dm._canonical_sha256({
        key: value for key, value in route.items() if key != "result_sha256"
    })
    _write_json(routed["route_path"], route)
    routed["route_artifact"]["sha256"] = _sha(routed["route_path"])
    routed["route_artifact"]["size_bytes"] = routed["route_path"].stat().st_size
    problems = dm._verify_trace_route_evidence(
        routed["row"], manifest=routed["manifest"], matrix=routed["matrix"],
        preflight=routed["preflight"], output_root=Path(
            routed["manifest"]["output_root"]
        ), label="cell_results[0]",
    )
    assert any("outcome/method combination is invalid" in problem for problem in problems)


def test_full_done_reference_walk_invokes_route_semantic_verifier(case, monkeypatch):
    manifest = _build(case)
    route_path = case["output_root"] / "portable_lcd/cell_a/trace_budget_route.json"
    _write_json(route_path, {"placeholder": True})
    manifest["cell_results"][0]["artifacts"].append({
        "artifact_kind": "trace_route_record",
        "relative_path": route_path.relative_to(case["output_root"]).as_posix(),
        "sha256": _sha(route_path),
        "size_bytes": route_path.stat().st_size,
    })
    calls = []

    def semantic(*args, **kwargs):
        calls.append((args, kwargs))
        return ["sentinel routed semantic failure"]

    monkeypatch.setattr(dm, "_verify_trace_route_evidence", semantic)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert len(calls) == 1
    assert "sentinel routed semantic failure" in problems


def test_tracked_reference_can_be_rehashed_at_sealed_commit(case):
    source_bytes = (case["repo"] / "README").read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    (case["repo"] / "README").write_text("later source\n")
    _git(case["repo"], "add", "README")
    _git(case["repo"], "commit", "-qm", "later")
    payload, observed_sha, origin = dm._reference_bytes_at_seal(
        case["repo"] / "README",
        expected_sha256=source_sha,
        repo_root=case["repo"],
        source_commit=case["head"],
    )
    assert payload == source_bytes
    assert observed_sha == source_sha
    assert origin == "sealed_commit"


@pytest.mark.campaign_evidence
def test_only_exact_bootstrap_relocates_former_checkout_to_sealed_git_bytes():
    manifest = json.loads(Path(
        "/home/jakeshea/sm86_reference_capture_r27/chunks/"
        "portable_lcd-000-bootstrap/DONE.json"
    ).read_text())
    legacy = (
        dm.HISTORICAL_BOOTSTRAP_REPOSITORY_ROOT
        / "data/local_labels/campaign/trace_retention_budget.json"
    )
    mapped = dm._historical_bootstrap_repo_reference(
        legacy, manifest=manifest, repo_root=_REPO,
    )
    assert mapped == _REPO / "data/local_labels/campaign/trace_retention_budget.json"
    payload, observed, origin = dm._reference_bytes_at_seal(
        mapped,
        expected_sha256=manifest["trace_budget_sha256"],
        repo_root=_REPO,
        source_commit=manifest["source_commit_pre"],
    )
    assert hashlib.sha256(payload).hexdigest() == observed == (
        manifest["trace_budget_sha256"]
    )
    assert origin == "sealed_commit"

    altered = copy.deepcopy(manifest)
    altered["expected_cells"] += 1
    assert dm._historical_bootstrap_repo_reference(
        legacy, manifest=altered, repo_root=_REPO,
    ) == legacy


def test_canonical_membership_hash_is_order_independent(case):
    members = case["kwargs"]["chunk_membership"]
    assert dm.chunk_membership_sha256(members) == dm.chunk_membership_sha256(
        list(reversed(members)))


@pytest.mark.parametrize("bad", ["", "HEAD", "f541a83", "a" * 40 + " "])
def test_noncanonical_source_commit_refuses_seal(case, bad):
    with pytest.raises(dm.DoneManifestError, match="commit|HEAD"):
        _build(case, source_commit_pre=bad, source_commit_post=bad,
               source_head_at_seal=bad)


def test_source_commit_drift_refuses_seal(case):
    with pytest.raises(dm.DoneManifestError, match="drift"):
        _build(case, source_commit_post="c" * 40)


def test_caller_claiming_dirty_tree_refuses_seal(case):
    with pytest.raises(dm.DoneManifestError, match="dirty chunks"):
        _build(case, working_tree_clean=False)


def test_actual_dirty_tree_refuses_seal_even_when_caller_claims_clean(case):
    (case["repo"] / "untracked").write_text("dirty\n")
    with pytest.raises(dm.DoneManifestError, match="working tree is dirty"):
        _build(case)


def test_zero_cell_vacuous_seal_is_impossible(case):
    with pytest.raises(dm.DoneManifestError, match="nonempty|positive"):
        _build(case, expected_cells=0, admitted=0,
               chunk_membership=[], cell_results=[])


def test_duplicate_chunk_identity_refuses_seal(case):
    members = copy.deepcopy(case["kwargs"]["chunk_membership"])
    members.append(copy.deepcopy(members[0]))
    with pytest.raises(dm.DoneManifestError, match="duplicate chunk authority"):
        _build(case, chunk_membership=members, expected_cells=3)


def test_duplicate_cell_result_identity_refuses_seal(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    duplicate = copy.deepcopy(results[0])
    duplicate["detail"] = "same identity, different JSON"
    results.append(duplicate)
    with pytest.raises(dm.DoneManifestError, match="duplicate cell-result"):
        _build(case, cell_results=results, admitted=3)


def test_missing_cell_result_refuses_seal(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"][:-1])
    with pytest.raises(dm.DoneManifestError, match="membership/result mismatch"):
        _build(case, cell_results=results, admitted=1)


def test_extra_cell_result_refuses_seal(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    extra = copy.deepcopy(results[0])
    extra["slug"] = "not_in_chunk"
    results.append(extra)
    with pytest.raises(dm.DoneManifestError, match="membership/result mismatch"):
        _build(case, cell_results=results, admitted=3)


def test_result_authority_hash_must_match_membership(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    results[0]["authority_cell_sha256"] = "f" * 64
    with pytest.raises(dm.DoneManifestError, match="authority hash differs"):
        _build(case, cell_results=results)


def test_closed_failure_reason_refuses_catchall(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    results[0].update(
        disposition="failed", typed_reason="unexpected_error", artifacts=[])
    with pytest.raises(dm.DoneManifestError, match="closed vocabulary"):
        _build(case, cell_results=results, admitted=1,
               typed_failures=None)


def test_admitted_cell_cannot_seal_without_complete_artifact_contract(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    results[0]["artifacts"] = [
        row for row in results[0]["artifacts"]
        if row["artifact_kind"] != "control_bits"
    ]
    with pytest.raises(dm.DoneManifestError, match="control_bits"):
        _build(case, cell_results=results)


def test_terminal_trace_requires_exactly_one_ncu_launch_observation(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    results[0]["artifacts"] = [
        row for row in results[0]["artifacts"]
        if row["artifact_kind"] != "ncu_launch_observation"
    ]
    with pytest.raises(dm.DoneManifestError, match="ncu_launch_observation"):
        _build(case, cell_results=results)


def test_terminal_trace_rejects_duplicate_ncu_launch_observations(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    _append_artifact(
        case,
        results[0],
        kind="ncu_launch_observation",
        name="duplicate_ncu_launch_observation.json",
    )
    with pytest.raises(
        dm.DoneManifestError,
        match="requires exactly one ncu_launch_observation",
    ):
        _build(case, cell_results=results)


def test_legacy_count_is_an_assertion_not_an_independent_truth(case):
    with pytest.raises(dm.DoneManifestError, match="derived admitted"):
        _build(case, admitted=1)


def test_artifact_parent_escape_refuses_seal(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    results[0]["artifacts"][0]["relative_path"] = "../outside.json"
    with pytest.raises(dm.DoneManifestError, match="beneath output_root"):
        _build(case, cell_results=results)


def test_zero_byte_execution_log_is_still_hash_bound(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    rel = "portable_lcd/cell_a/stdout.log"
    path = case["output_root"] / rel
    path.write_bytes(b"")
    results[0]["artifacts"].append({
        "artifact_kind": "execution_log",
        "relative_path": rel,
        "sha256": _sha(path),
        "size_bytes": 0,
    })
    manifest = _build(case, cell_results=results)
    assert dm.verify_seal(manifest, repo_root=case["repo"]) == []


def test_staging_root_override_supports_atomic_publish(case):
    staging = case["output_root"]
    final = staging.parent / "published_chunk"
    manifest = _build(
        case,
        output_root=str(final),
        verification_output_root=staging,
    )
    assert not final.exists()
    staging.rename(final)
    assert dm.verify_seal(manifest, repo_root=case["repo"]) == []


def test_artifact_tamper_invalidates_resume(case):
    manifest = _build(case)
    artifact = manifest["cell_results"][0]["artifacts"][0]
    path = case["output_root"] / artifact["relative_path"]
    path.write_bytes(b"tampered artifact\n")
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("artifact sha256 mismatch" in problem for problem in problems)


def test_unlisted_trace_payload_invalidates_resume(case):
    manifest = _build(case)
    extra = case["output_root"] / "portable_lcd" / "cell_a" / "threadblocks" / "extra.pb"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"consumer-visible but unsealed")
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("unlisted artifacts" in problem for problem in problems)


def test_zero_byte_semantic_payload_cannot_seal(case):
    kwargs = copy.deepcopy(case["kwargs"])
    artifact = next(
        row for row in kwargs["cell_results"][0]["artifacts"]
        if row["artifact_kind"] == "trace_threadblock"
    )
    path = case["output_root"] / artifact["relative_path"]
    path.write_bytes(b"")
    artifact["sha256"] = _sha(path)
    artifact["size_bytes"] = 0
    with pytest.raises(dm.DoneManifestError, match="must be positive"):
        dm.build_done_manifest(**kwargs)


def test_artifact_symlink_escape_invalidates_resume(case):
    manifest = _build(case)
    artifact = manifest["cell_results"][0]["artifacts"][0]
    path = case["output_root"] / artifact["relative_path"]
    outside = case["output_root"].parent / "outside"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("escapes output_root" in problem for problem in problems)


def test_stale_input_hash_invalidates_resume(case):
    manifest = _build(case)
    pins_path = case["paths"]["pins"]
    pins = json.loads(pins_path.read_text())
    pins["new_field"] = "tamper"
    _write_json(pins_path, pins)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("stale pins hash" in problem for problem in problems)


def test_matrix_missing_declared_cell_invalidates_resume(case):
    manifest = _build(case)
    matrix_path = case["paths"]["capture_matrix"]
    matrix = json.loads(matrix_path.read_text())
    matrix["chunks"][0]["cells"].pop()
    _write_json(matrix_path, matrix)
    manifest["capture_matrix_sha256"] = _sha(matrix_path)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("outside capture-matrix chunk" in problem for problem in problems)


def test_matrix_without_chunks_invalidates_resume(case):
    manifest = _build(case)
    matrix_path = case["paths"]["capture_matrix"]
    matrix = json.loads(matrix_path.read_text())
    matrix.pop("chunks")
    _write_json(matrix_path, matrix)
    manifest["capture_matrix_sha256"] = _sha(matrix_path)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("lacks required nonempty" in problem for problem in problems)


def test_matrix_full_row_authority_hash_is_cross_checked(case):
    manifest = _build(case)
    matrix_path = case["paths"]["capture_matrix"]
    matrix = json.loads(matrix_path.read_text())
    matrix["cells"][0]["authority_cell_sha256"] = "f" * 64
    _write_json(matrix_path, matrix)
    manifest["capture_matrix_sha256"] = _sha(matrix_path)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("capture matrix authority hash mismatch" in p for p in problems)


def test_authority_row_tamper_invalidates_cell_hash(case):
    manifest = _build(case)
    authority_path = case["paths"]["roster_authority"]
    authority = json.loads(authority_path.read_text())
    authority["cells"][0]["binary"] = "different/binary"
    _write_json(authority_path, authority)
    manifest["roster_authority_sha256"] = _sha(authority_path)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("authority row hash mismatch" in problem for problem in problems)


def test_duplicate_authority_identity_invalidates_resume(case):
    manifest = _build(case)
    authority_path = case["paths"]["roster_authority"]
    authority = json.loads(authority_path.read_text())
    authority["cells"].append(copy.deepcopy(authority["cells"][0]))
    _write_json(authority_path, authority)
    manifest["roster_authority_sha256"] = _sha(authority_path)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("duplicate identities" in problem for problem in problems)


def test_preflight_must_match_source_and_admit_capture(case):
    manifest = _build(case)
    preflight_path = case["paths"]["preflight"]
    preflight = json.loads(preflight_path.read_text())
    preflight["git_head"] = "d" * 40
    preflight["capture_admitted"] = False
    _write_json(preflight_path, preflight)
    manifest["preflight_sha256"] = _sha(preflight_path)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("did not admit" in problem for problem in problems)
    assert any("preflight git_head" in problem for problem in problems)


def test_current_head_change_invalidates_resume(case):
    manifest = _build(case)
    (case["repo"] / "next").write_text("next\n")
    _git(case["repo"], "add", "next")
    _git(case["repo"], "commit", "-qm", "next")
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("current git HEAD" in problem for problem in problems)


def test_current_dirty_tree_invalidates_resume(case):
    manifest = _build(case)
    (case["repo"] / "dirty").write_text("dirty\n")
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any("working tree is dirty" in problem for problem in problems)


def test_schema_and_python_closed_enums_are_lockstep():
    assert set(_SCHEMA["$defs"]["rosterSource"]["enum"]) == set(dm.ROSTER_SOURCES)
    assert set(_SCHEMA["$defs"]["typedReason"]["enum"]) == set(el.REASON_VOCABULARY)
    assert set(_SCHEMA["$defs"]["artifact"]["properties"]
               ["artifact_kind"]["enum"]) == set(dm.ARTIFACT_KINDS)
    disposition_enum = (_SCHEMA["$defs"]["cellResult"]["properties"]
                        ["disposition"]["enum"])
    assert set(disposition_enum) == set(dm.CELL_DISPOSITIONS)
    attestation_schema = _SCHEMA["$defs"]["definitionEquivalenceAttestation"]
    assert attestation_schema["properties"]["rule_id"]["const"] == (
        dm.SOL_DEFINITION_EQUIVALENCE_RULE_ID
    )
    assert set(attestation_schema["required"]) == {
        "rule_id", "result", "source_definition_sha256",
        "staged_definition_sha256", "top_level_field_states",
        "axes_container_state", "axis_description_states",
    }
    assert attestation_schema["oneOf"] == [
        {
            "required": [
                "tensor_container_states", "tensor_description_states",
            ],
        },
        {
            "not": {
                "anyOf": [
                    {"required": ["tensor_container_states"]},
                    {"required": ["tensor_description_states"]},
                ],
            },
        },
    ]


@pytest.mark.campaign_evidence
def test_e387_legacy_definition_context_hashes_exact_git_blobs():
    assert dm.SOL_E387_LEGACY_SOURCE_COMMIT == sol_def.E387_LEGACY_SOURCE_COMMIT
    assert dm.SOL_E387_LEGACY_COMPARATOR_SHA256 == (
        sol_def.E387_LEGACY_COMPARATOR_SHA256
    )
    assert dm.SOL_E387_LEGACY_CAPTURE_ADAPTER_SHA256 == (
        sol_def.E387_LEGACY_CAPTURE_ADAPTER_SHA256
    )
    prepare = {
        "artifacts": {
            "adapter_source": {
                "path": (
                    "/historical/checkout/"
                    + dm.SOL_E387_LEGACY_CAPTURE_ADAPTER_PATH.as_posix()
                ),
                "sha256": dm.SOL_E387_LEGACY_CAPTURE_ADAPTER_SHA256,
            },
        },
    }
    problems = []
    context = dm._e387_legacy_definition_code_context(
        prepare_manifest=prepare,
        repo_root=_REPO,
        source_commit=dm.SOL_E387_LEGACY_SOURCE_COMMIT,
        label="fixture",
        problems=problems,
    )
    assert problems == []
    assert context == {
        "source_commit": dm.SOL_E387_LEGACY_SOURCE_COMMIT,
        "comparator_sha256": dm.SOL_E387_LEGACY_COMPARATOR_SHA256,
        "capture_adapter_sha256": (
            dm.SOL_E387_LEGACY_CAPTURE_ADAPTER_SHA256
        ),
    }

    prepare["artifacts"]["adapter_source"]["sha256"] = "0" * 64
    problems = []
    assert dm._e387_legacy_definition_code_context(
        prepare_manifest=prepare,
        repo_root=_REPO,
        source_commit=dm.SOL_E387_LEGACY_SOURCE_COMMIT,
        label="fixture",
        problems=problems,
    ) is None
    assert any("exact e387 adapter" in problem for problem in problems)


def test_schema_rejects_vacuous_zero_cell_document(case):
    manifest = _build(case)
    manifest["expected_cells"] = 0
    manifest["chunk_membership"] = []
    manifest["cell_results"] = []
    manifest["counts"] = {"admitted": 0, "excluded": 0, "failed": 0}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, _SCHEMA)


def test_round28_cell_requires_preparation_failure_backrefs(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    results[0].pop("prior_preparation_failures")
    with pytest.raises(dm.DoneManifestError, match="prior_preparation_failures"):
        _build(case, cell_results=results)


def test_schema_requires_preparation_failure_backrefs(case):
    manifest = _build(case)
    manifest["cell_results"][0].pop("prior_preparation_failures")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, _SCHEMA)


def test_cell_result_preparation_backrefs_must_equal_done_row(case):
    manifest = _build(case)
    _row, artifact, path, document = _cell_result_document(case, manifest)
    document["prior_preparation_failures"] = [{
        "ledger_entry_id": "00000000-0000-4000-8000-000000000001",
        "ledger_path": str(
            case["output_root"].parent / "preparation_failures"
            / "preparation_failure_ledger.jsonl"
        ),
        "archive_path": str(
            case["output_root"].parent / "preparation_failures" / "attempt"
        ),
    }]
    _write_and_reseal(path, artifact, document)
    problems = dm.verify_seal(manifest, repo_root=case["repo"])
    assert any(
        "cell result provenance mismatch" in problem
        and "prior_preparation_failures" in problem
        for problem in problems
    )


def test_preparation_archive_artifact_path_cannot_publish(case):
    results = copy.deepcopy(case["kwargs"]["cell_results"])
    results[0]["artifacts"][0]["relative_path"] = (
        "preparation_failures/attempt/forensic.json"
    )
    with pytest.raises(dm.DoneManifestError, match="PREPARATION_ARCHIVE_LEAK"):
        _build(case, cell_results=results)


def _sealed_preparation_closure_fixture(tmp_path: Path) -> tuple[dict, dict, Path]:
    from tools.local_labels import archive_preparation_failure as ap
    from tools.local_labels.test_archive_preparation_failure import (
        _fixture as archive_fixture,
    )

    request = archive_fixture(
        tmp_path / "archive-fixture",
        prep_root=tmp_path / "capture" / "preparation_failures",
    )
    entry = ap.archive_preparation_failure(request)
    ledger_path = (
        request.preparation_failures_root / ap.PREPARATION_FAILURE_LEDGER_NAME
    ).resolve()
    entries = ap.load_preparation_failure_ledger(
        ledger_path, verify_archives=True,
    )
    refs = ap.preparation_failure_backrefs(
        entries, cell_id=request.cell_id, ledger_path=ledger_path,
    )
    source, slug = request.cell_id.split("__", 1)
    row = {
        "roster_source": source,
        "slug": slug,
        "prior_preparation_failures": refs,
    }
    manifest = {
        "chunk_id": request.chunk_id,
        "output_root": str(tmp_path / "capture" / "chunks" / request.chunk_id),
        "cell_results": [row],
    }
    return manifest, entry, ledger_path


def test_preparation_failure_ledger_closes_bidirectionally(tmp_path: Path):
    manifest, _entry, _ledger = _sealed_preparation_closure_fixture(tmp_path)
    row = manifest["cell_results"][0]
    assert dm.verify_manifest_preparation_failure_closure(
        manifest, result_documents=[(row, copy.deepcopy(row))],
    ) == []


def test_preparation_failure_backref_set_mismatch_is_typed(tmp_path: Path):
    manifest, _entry, _ledger = _sealed_preparation_closure_fixture(tmp_path)
    row = manifest["cell_results"][0]
    row["prior_preparation_failures"] = []
    problems = dm.verify_manifest_preparation_failure_closure(
        manifest, result_documents=[(row, copy.deepcopy(row))],
    )
    assert any("PREPARATION_LEDGER_INCONSISTENT" in problem for problem in problems)


def test_preparation_failure_inventory_tamper_is_typed(tmp_path: Path):
    manifest, entry, _ledger = _sealed_preparation_closure_fixture(tmp_path)
    inventory_path = Path(entry["inventory_file"])
    inventory_path.write_bytes(inventory_path.read_bytes() + b" ")
    problems = dm.verify_manifest_preparation_failure_closure(manifest)
    assert any("PREPARATION_LEDGER_INCONSISTENT" in problem for problem in problems)


def test_preparation_archive_without_ledger_is_typed(tmp_path: Path):
    manifest, entry, ledger_path = _sealed_preparation_closure_fixture(tmp_path)
    ledger_path.unlink()
    problems = dm.verify_manifest_preparation_failure_closure(manifest)
    assert problems[0] == (
        "PREPARATION_LEDGER_INCONSISTENT: archive evidence exists without "
        f"ledger: {[entry['archive_path']]!r}"
    )


@pytest.mark.parametrize("root_exists", [False, True])
def test_preparation_closure_admits_missing_or_empty_global_root(
    tmp_path: Path, root_exists: bool,
):
    chunk_id = "portable_lcd-001"
    campaign_root = tmp_path / "capture"
    if root_exists:
        (campaign_root / dm.PREPARATION_FAILURE_ROOT_NAME).mkdir(parents=True)
    manifest = {
        "chunk_id": chunk_id,
        "output_root": str(campaign_root / "chunks" / chunk_id),
        "cell_results": [{
            "roster_source": "portable_lcd",
            "slug": "no-preparation-attempt",
            "prior_preparation_failures": [],
        }],
    }
    assert dm.verify_manifest_preparation_failure_closure(manifest) == []


def test_semantic_verifier_catches_duplicate_identity_schema_cannot(case):
    manifest = _build(case)
    duplicate = copy.deepcopy(manifest["cell_results"][0])
    duplicate["detail"] = "different object, same authority identity"
    manifest["cell_results"].append(duplicate)
    # uniqueItems cannot see identity-level duplication when other fields differ.
    jsonschema.validate(manifest, _SCHEMA)
    problems = dm.verify_seal(
        manifest, repo_root=case["repo"], rehash_references=False,
        require_current_head=False, require_clean_tree=False,
    )
    assert any("duplicate cell-result" in problem for problem in problems)


def test_verifier_returns_problems_instead_of_throwing_on_malformed_rows():
    malformed = {
        "cell_results": "not-an-array",
        "chunk_membership": [None],
        "input_paths": {},
    }
    problems = dm.verify_seal(
        malformed, repo_root=Path("/definitely/missing"),
        rehash_references=False, require_current_head=False,
        require_clean_tree=False,
    )
    assert any("schema violation" in problem for problem in problems)
    assert any("chunk_membership" in problem for problem in problems)


def test_cli_seals_from_matrix_and_rehashes_everything(case, monkeypatch):
    cells_path = case["inputs"] / "cells.json"
    admission_path = case["inputs"] / "admission.json"
    _write_json(cells_path, case["kwargs"]["cell_results"])
    _write_json(admission_path, case["kwargs"]["admission"])
    out = case["output_root"] / "DONE.json"
    monkeypatch.setattr(dm, "REPO_ROOT", case["repo"])
    rc = dm.main([
        "--command", "capture --chunk chunk_3",
        "--source-commit-pre", case["head"],
        "--pins", str(case["paths"]["pins"]),
        "--roster-authority", str(case["paths"]["roster_authority"]),
        "--trace-budget", str(case["paths"]["trace_budget"]),
        "--capture-matrix", str(case["paths"]["capture_matrix"]),
        "--preflight", str(case["paths"]["preflight"]),
        "--admission", str(admission_path),
        "--output-root", str(case["output_root"]),
        "--cells", str(cells_path),
        "--chunk-id", "chunk_3",
        "--out", str(out),
    ])
    assert rc == 0
    manifest = json.loads(out.read_text())
    jsonschema.validate(manifest, _SCHEMA)
    assert dm.verify_seal(manifest, repo_root=case["repo"]) == []


def test_cli_dirty_tree_refuses_and_writes_no_done(case, monkeypatch):
    cells_path = case["inputs"] / "cells.json"
    admission_path = case["inputs"] / "admission.json"
    _write_json(cells_path, case["kwargs"]["cell_results"])
    _write_json(admission_path, case["kwargs"]["admission"])
    (case["repo"] / "dirty").write_text("dirty\n")
    out = case["output_root"] / "DONE.json"
    monkeypatch.setattr(dm, "REPO_ROOT", case["repo"])
    rc = dm.main([
        "--command", "capture --chunk chunk_3",
        "--source-commit-pre", case["head"],
        "--pins", str(case["paths"]["pins"]),
        "--roster-authority", str(case["paths"]["roster_authority"]),
        "--trace-budget", str(case["paths"]["trace_budget"]),
        "--capture-matrix", str(case["paths"]["capture_matrix"]),
        "--preflight", str(case["paths"]["preflight"]),
        "--admission", str(admission_path),
        "--output-root", str(case["output_root"]),
        "--cells", str(cells_path),
        "--chunk-id", "chunk_3",
        "--out", str(out),
    ])
    assert rc == 1
    assert not out.exists()
