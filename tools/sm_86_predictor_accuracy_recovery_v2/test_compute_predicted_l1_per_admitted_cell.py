from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings
from tools.sm_86_predictor_accuracy_recovery_v2 import (
    compute_predicted_l1_per_admitted_cell as predictor,
)
from tools.sm_86_predictor_accuracy_recovery_v2 import (
    compute_warp_pool_l1_per_admitted_cell as warp_pool,
)


def _identity_row() -> dict[str, str]:
    return {
        "arch": "sm_86",
        "binary_sha256": "1" * 64,
        "source_commit": "2" * 40,
        "kernel_signature_sha256": "3" * 64,
        "ncu_metric_version": "v5",
    }


def _install_inputs(
    root: Path,
    *,
    slug: str = "cell",
    shape_point: str = "shape",
) -> tuple[Path, Path, Path]:
    manifest_path = (
        root
        / "data/local_labels/cache_op_classifier_manifests/sm_86"
        / f"{slug}__{shape_point}.json"
    )
    manifest_path.parent.mkdir(parents=True)
    trace_dir = (
        root
        / "workloads/simbench_bw/results"
        / slug
        / "rtx_3090_ti/sm_86"
        / f"trace_{shape_point}"
    )
    threadblock_path = (
        trace_dir
        / "threadblocks/device_0/stream_0/kernel_1/block_0.pb"
    )
    threadblock_path.parent.mkdir(parents=True)
    return manifest_path, trace_dir, threadblock_path


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_public_build_artifact_does_not_restamp_unauthenticated_prior_scalar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor, "_REPO_ROOT", tmp_path)
    identity = _identity_row()
    label_path = tmp_path / "label_admitted.json"
    label_path.write_text(
        json.dumps({
            "cells": [{
                "slug": "cell",
                "shape_point": "shape",
                **identity,
            }],
        }),
        encoding="utf-8",
    )
    prior_path = tmp_path / "predicted_l1.json"
    prior_path.write_text(
        json.dumps({
            "cells": [{
                "slug": "cell",
                "shape_point": "shape",
                "arch": "sm_86",
                "binary_sha256": "a" * 64,
                "source_commit": "b" * 40,
                "kernel_signature_sha256": "c" * 64,
                "ncu_metric_version": "stale-v4",
                "cache_layer": "L1TEX",
                "result_kind": "scalar",
                "predicted_hit_rate": 0.875,
                "typed_reason": None,
                "evidence": {"unauthenticated_prior": True},
                "input_digests": {
                    "classifier_manifest_sha256": "d" * 64,
                    "dynamic_trace_pb_sha256": "e" * 64,
                    "threadblock_proto_set_sha256": "f" * 64,
                },
            }],
        }),
        encoding="utf-8",
    )

    artifact = predictor.build_artifact(
        label_admitted_path=label_path,
        existing_artifact_path=prior_path,
    )

    assert artifact["cohort_size"] == 1
    row = artifact["cells"][0]
    assert row["result_kind"] == "typed_unavailable"
    assert row["predicted_hit_rate"] is None
    assert row["typed_reason"] == "predictor_l1_sass_provenance_missing"
    assert row["binary_sha256"] == identity["binary_sha256"]
    assert row["input_digests"] == {
        "classifier_manifest_sha256": _sha(b""),
        "dynamic_trace_pb_sha256": _sha(b""),
        "threadblock_proto_set_sha256": _sha(b""),
    }
    assert "preserved_from_prior_artifact" not in row["evidence"]
    assert row["evidence"].get("unauthenticated_prior") is None


def test_compute_uses_one_retained_generation_for_prediction_and_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor, "_REPO_ROOT", tmp_path)
    manifest_path, trace_dir, threadblock_path = _install_inputs(tmp_path)
    old_manifest = (
        b'{"sass_provenance_status":"matched","generation":"old"}\n'
    )
    old_root = b"old-root-protobuf"
    old_threadblock = b"old-threadblock-protobuf"
    manifest_path.write_bytes(old_manifest)
    (trace_dir / "dynamic_trace.pb").write_bytes(old_root)
    threadblock_path.write_bytes(old_threadblock)

    read_counts: dict[str, int] = {}
    original_read_snapshot = predictor.immutable.PinnedDirectoryReader.read_snapshot

    def count_snapshot(self, relative_path: str, **kwargs):
        read_counts[relative_path] = read_counts.get(relative_path, 0) + 1
        return original_read_snapshot(self, relative_path, **kwargs)

    monkeypatch.setattr(
        predictor.immutable.PinnedDirectoryReader,
        "read_snapshot",
        count_snapshot,
    )

    new_manifest = (
        b'{"sass_provenance_status":"matched","generation":"new"}\n'
    )
    new_root = b"new-root-protobuf"
    new_threadblock = b"new-threadblock-protobuf"

    def predict_after_generation_swap(**kwargs):
        assert kwargs["classifier_manifest"]["generation"] == "old"
        assert kwargs["retained_trace_pb_payload"] == old_root
        assert kwargs["retained_threadblock_pb_payloads"] == [old_threadblock]

        replacement_manifest = manifest_path.with_name("replacement.json")
        replacement_manifest.write_bytes(new_manifest)
        os.replace(replacement_manifest, manifest_path)

        retired_trace_dir = trace_dir.with_name(f"{trace_dir.name}.retired")
        trace_dir.rename(retired_trace_dir)
        replacement_threadblock = (
            trace_dir
            / "threadblocks/device_0/stream_0/kernel_1/block_0.pb"
        )
        replacement_threadblock.parent.mkdir(parents=True)
        (trace_dir / "dynamic_trace.pb").write_bytes(new_root)
        replacement_threadblock.write_bytes(new_threadblock)

        return {
            "result_kind": "scalar",
            "cache_layer": "L1TEX",
            "predicted_hit_rate": 0.25,
            "typed_reason": None,
            "evidence": {"consumed_generation": "old"},
        }

    monkeypatch.setattr(
        predictor.lph,
        "predict_l1_from_trace_dir",
        predict_after_generation_swap,
    )

    entry = predictor.compute_for_cell(
        slug="cell",
        shape_point="shape",
        cell_row=_identity_row(),
    )

    expected_threadblock_set = _sha(
        (
            f"{threadblock_path.relative_to(tmp_path)}\t"
            f"{_sha(old_threadblock)}\n"
        ).encode("utf-8")
    )
    assert entry["result_kind"] == "scalar"
    assert entry["evidence"]["consumed_generation"] == "old"
    assert entry["input_digests"] == {
        "classifier_manifest_sha256": _sha(old_manifest),
        "dynamic_trace_pb_sha256": _sha(old_root),
        "threadblock_proto_set_sha256": expected_threadblock_set,
    }
    assert entry["input_digests"]["classifier_manifest_sha256"] != _sha(
        new_manifest
    )
    assert entry["input_digests"]["dynamic_trace_pb_sha256"] != _sha(new_root)
    assert read_counts == {
        "cell__shape.json": 1,
        "dynamic_trace.pb": 1,
        "threadblocks/device_0/stream_0/kernel_1/block_0.pb": 1,
    }


def test_trace_mutation_during_snapshot_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor, "_REPO_ROOT", tmp_path)
    manifest_path, trace_dir, threadblock_path = _install_inputs(tmp_path)
    manifest_path.write_bytes(b'{"sass_provenance_status":"matched"}\n')
    (trace_dir / "dynamic_trace.pb").write_bytes(b"root")
    threadblock_path.write_bytes(b"threadblock-old")

    original_read_snapshot = predictor.immutable.PinnedDirectoryReader.read_snapshot
    swapped = False

    def swap_after_root(self, relative_path: str, **kwargs):
        nonlocal swapped
        snapshot = original_read_snapshot(self, relative_path, **kwargs)
        if relative_path == "dynamic_trace.pb" and not swapped:
            replacement = threadblock_path.with_name("replacement.pb")
            replacement.write_bytes(b"threadblock-new")
            os.replace(replacement, threadblock_path)
            swapped = True
        return snapshot

    monkeypatch.setattr(
        predictor.immutable.PinnedDirectoryReader,
        "read_snapshot",
        swap_after_root,
    )
    monkeypatch.setattr(
        predictor.lph,
        "predict_l1_from_trace_dir",
        lambda **_kwargs: pytest.fail("prediction must not run after mutation"),
    )

    with pytest.raises(
        predictor.immutable.ImmutableArtifactError,
        match="(generation changed|mutated|changed during use)",
    ):
        predictor.compute_for_cell(
            slug="cell",
            shape_point="shape",
            cell_row=_identity_row(),
        )


def test_trace_snapshot_enforces_per_file_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor, "_REPO_ROOT", tmp_path)
    _manifest_path, trace_dir, threadblock_path = _install_inputs(tmp_path)
    (trace_dir / "dynamic_trace.pb").write_bytes(b"too-large")
    threadblock_path.write_bytes(b"x")
    monkeypatch.setattr(
        predictor,
        "_TRACE_INPUT_LIMITS",
        dataclasses.replace(
            predictor._TRACE_INPUT_LIMITS,
            trace_root_bytes=1,
            threadblock_bytes=1,
        ),
    )

    with pytest.raises(predictor.immutable.ImmutableArtifactLimitError):
        predictor._snapshot_trace_inputs(trace_dir)


def test_l1_helper_parsers_use_retained_bytes_without_reopening_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_payload = b"retained-root"
    threadblock_payload = b"retained-threadblock"
    preflighted: list[bytes] = []
    parsed: list[bytes] = []

    monkeypatch.setattr(
        predictor.lph,
        "preflight_trace_protobuf",
        lambda payload: preflighted.append(payload),
    )
    monkeypatch.setattr(
        predictor.lph,
        "preflight_threadblock_protobuf",
        lambda payload: preflighted.append(payload),
    )

    class FakeTrace:
        trace_contract_id = "sm86_addr_carry_shmem_split_v2"
        memory_payload_classes = (1, 2, 3, 6, 8)
        gpu_device: dict = {}

        def ParseFromString(self, payload: bytes) -> None:
            parsed.append(payload)

    class FakeTraceModule:
        Trace = FakeTrace

    assert predictor.lph._find_kernel_record(
        FakeTraceModule,
        tmp_path / "missing-root.pb",
        1,
        retained_trace_pb_payload=root_payload,
    ) is None

    class FakeBlockId:
        x = 0
        y = 0
        z = 0

    class FakeThreadblock:
        block_id = FakeBlockId()
        warps: dict = {}

        def ParseFromString(self, payload: bytes) -> None:
            parsed.append(payload)

        def HasField(self, field: str) -> bool:
            return field == "block_id"

    class FakeThreadblockModule:
        @staticmethod
        def threadblock() -> FakeThreadblock:
            return FakeThreadblock()

    assert predictor.lph._walk_threadblock_protos(
        FakeThreadblockModule,
        [tmp_path / "missing-threadblock.pb"],
        object(),
        grid_x=1,
        grid_y=1,
        grid_dim=1,
        retained_threadblock_pb_payloads=[threadblock_payload],
    ) == []
    assert preflighted == [root_payload, threadblock_payload]
    assert parsed == [root_payload, threadblock_payload]


def _install_warp_pool_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
    capabilities: tuple[int, ...],
    include_threadblock: bool,
) -> tuple[Path, Path]:
    monkeypatch.setattr(warp_pool, "_REPO_ROOT", tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"sass_provenance_status": status}),
        encoding="utf-8",
    )
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    trace_pb_path = trace_dir / "dynamic_trace.pb"
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2
    trace = trace_pb2.Trace()
    trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2"
    trace.memory_payload_classes.extend(capabilities)
    kernel = trace.gpu_device[0].streams[0].kernels.add()
    kernel.id = 1
    kernel.name = "fixture_kernel___0"
    kernel.grid_dim.x = kernel.grid_dim.y = kernel.grid_dim.z = 1
    kernel.block_dim.x = 32
    kernel.block_dim.y = kernel.block_dim.z = 1
    kernel.size_shared_memory = 0
    kernel.shared_memory_static_bytes = 0
    kernel.shared_memory_dynamic_bytes = 0
    trace_pb_path.write_bytes(trace.SerializeToString())

    threadblock_path = (
        trace_dir
        / "threadblocks/device_0/stream_0/kernel_1/block_0.pb"
    )
    if include_threadblock:
        threadblock_path.parent.mkdir(parents=True)
        threadblock_path.write_bytes(b"unused-by-regression")
    monkeypatch.setattr(
        warp_pool,
        "_manifest_path_for",
        lambda _slug, _shape: manifest_path,
    )
    monkeypatch.setattr(
        warp_pool,
        "_trace_dir_for",
        lambda _slug, _shape: trace_dir,
    )
    return trace_pb_path, threadblock_path


def test_warp_pool_no_load_rejects_duplicate_trace_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_warp_pool_cell(
        tmp_path,
        monkeypatch,
        status="matched_no_load_class_instructions",
        capabilities=(1, 1, 2, 3, 6, 8),
        include_threadblock=False,
    )

    with pytest.raises(
        predictor.lph.L1ProductionHelperTraceCapabilitiesInvalid,
        match="trace capabilities invalid",
    ):
        warp_pool.compute_for_cell(
            slug="cell",
            shape_point="shape",
            cell_row=_identity_row(),
        )


def test_valid_trace_reaches_warp_pool_scalar_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_warp_pool_cell(
        tmp_path,
        monkeypatch,
        status="matched",
        capabilities=(1, 2, 3, 6, 8),
        include_threadblock=True,
    )
    monkeypatch.setattr(
        warp_pool.lph,
        "_walk_threadblock_protos",
        lambda *_args, **_kwargs: [object()],
    )
    monkeypatch.setattr(
        warp_pool.lta,
        "build_per_warp_l1_accesses",
        lambda *_args, **_kwargs: [((0, 0), [object()])],
    )
    geometry = SimpleNamespace(
        capacity=SimpleNamespace(
            line_capacity=1,
            capacity_bytes=128,
            property_name="fixture_capacity",
        ),
        latencies=SimpleNamespace(hit_latency=1, miss_latency=2),
        mshr=SimpleNamespace(count=1, source="fixture_mshr"),
        replacement=SimpleNamespace(modeled="LRU", source="fixture_replacement"),
        set_mapping=SimpleNamespace(
            modeled="fully_associative",
            source="fixture_mapping",
        ),
    )
    monkeypatch.setattr(
        warp_pool,
        "resolve_walk_geometry",
        lambda **_kwargs: geometry,
    )
    walker_calls: list[tuple] = []

    def walk(*args, **kwargs):
        walker_calls.append((args, kwargs))
        return SimpleNamespace(
            hit_count=1,
            miss_count=0,
            bypass_count=0,
            mshr_merge_count=0,
            mshr_stall_events=0,
            line_fill_count=0,
            line_alloc_miss_count=0,
            sector_miss_resident_line_count=0,
        )

    monkeypatch.setattr(warp_pool, "walk_warp_pool", walk)

    entry = warp_pool.compute_for_cell(
        slug="cell",
        shape_point="shape",
        cell_row=_identity_row(),
    )

    assert entry["result_kind"] == "scalar"
    assert entry["predicted_hit_rate"] == 1.0
    assert entry["evidence"]["total_eligible_accesses"] == 1
    assert len(walker_calls) == 1


def _install_warp_pool_scalar_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        warp_pool.lta,
        "build_per_warp_l1_accesses",
        lambda *_args, **_kwargs: [((0, 0), [object()])],
    )
    geometry = SimpleNamespace(
        capacity=SimpleNamespace(
            line_capacity=1,
            capacity_bytes=128,
            property_name="fixture_capacity",
        ),
        latencies=SimpleNamespace(hit_latency=1, miss_latency=2),
        mshr=SimpleNamespace(count=1, source="fixture_mshr"),
        replacement=SimpleNamespace(
            modeled="LRU",
            source="fixture_replacement",
        ),
        set_mapping=SimpleNamespace(
            modeled="fully_associative",
            source="fixture_mapping",
        ),
    )
    monkeypatch.setattr(
        warp_pool,
        "resolve_walk_geometry",
        lambda **_kwargs: geometry,
    )
    monkeypatch.setattr(
        warp_pool,
        "walk_warp_pool",
        lambda *_args, **_kwargs: SimpleNamespace(
            hit_count=1,
            miss_count=0,
            bypass_count=0,
            mshr_merge_count=0,
            mshr_stall_events=0,
            line_fill_count=0,
            line_alloc_miss_count=0,
            sector_miss_resident_line_count=0,
        ),
    )


def test_warp_pool_uses_one_retained_generation_after_path_swaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_pb_path, threadblock_path = _install_warp_pool_cell(
        tmp_path,
        monkeypatch,
        status="matched",
        capabilities=(1, 2, 3, 6, 8),
        include_threadblock=True,
    )
    manifest_path = tmp_path / "manifest.json"
    old_manifest = json.dumps({
        "sass_provenance_status": "matched",
        "generation": "old",
    }).encode("utf-8")
    new_manifest = json.dumps({
        "sass_provenance_status": "matched",
        "generation": "new",
    }).encode("utf-8")
    manifest_path.write_bytes(old_manifest)
    old_root = trace_pb_path.read_bytes()
    old_threadblock = threadblock_path.read_bytes()
    new_root = b"replacement-root-must-not-be-opened"
    new_threadblock = b"replacement-threadblock-must-not-be-opened"

    read_counts: dict[str, int] = {}
    original_read_snapshot = (
        warp_pool.immutable.PinnedDirectoryReader.read_snapshot
    )

    def count_snapshot(self, relative_path: str, **kwargs):
        read_counts[relative_path] = read_counts.get(relative_path, 0) + 1
        return original_read_snapshot(self, relative_path, **kwargs)

    monkeypatch.setattr(
        warp_pool.immutable.PinnedDirectoryReader,
        "read_snapshot",
        count_snapshot,
    )

    original_find_kernel = warp_pool.lph._find_kernel_record
    swapped = False

    def find_kernel_after_swap(
        trace_pb2_mod,
        trace_path: Path,
        kernel_id: int,
        *,
        retained_trace_pb_payload: bytes | None = None,
    ):
        nonlocal swapped
        assert retained_trace_pb_payload == old_root
        if not swapped:
            replacement_manifest = manifest_path.with_name(
                "replacement-manifest.json"
            )
            replacement_manifest.write_bytes(new_manifest)
            os.replace(replacement_manifest, manifest_path)

            trace_dir = trace_pb_path.parent
            retired_trace_dir = trace_dir.with_name("trace-retired")
            trace_dir.rename(retired_trace_dir)
            replacement_threadblock = (
                trace_dir
                / "threadblocks/device_0/stream_0/kernel_1/block_0.pb"
            )
            replacement_threadblock.parent.mkdir(parents=True)
            (trace_dir / "dynamic_trace.pb").write_bytes(new_root)
            replacement_threadblock.write_bytes(new_threadblock)
            swapped = True
        return original_find_kernel(
            trace_pb2_mod,
            trace_path,
            kernel_id,
            retained_trace_pb_payload=retained_trace_pb_payload,
        )

    monkeypatch.setattr(
        warp_pool.lph,
        "_find_kernel_record",
        find_kernel_after_swap,
    )

    def walk_retained_threadblocks(
        _threadblock_pb2_mod,
        paths: list[Path],
        _import_trace_contract_mod,
        **kwargs,
    ) -> list[object]:
        assert paths == [threadblock_path]
        assert kwargs["retained_threadblock_pb_payloads"] == [
            old_threadblock
        ]
        return [object()]

    monkeypatch.setattr(
        warp_pool.lph,
        "_walk_threadblock_protos",
        walk_retained_threadblocks,
    )
    _install_warp_pool_scalar_stages(monkeypatch)

    entry = warp_pool.compute_for_cell(
        slug="cell",
        shape_point="shape",
        cell_row=_identity_row(),
    )

    expected_threadblock_set = _sha(
        (
            f"{threadblock_path.relative_to(tmp_path)}\t"
            f"{_sha(old_threadblock)}\n"
        ).encode("utf-8")
    )
    assert swapped
    assert entry["result_kind"] == "scalar"
    assert entry["input_digests"] == {
        "classifier_manifest_sha256": _sha(old_manifest),
        "dynamic_trace_pb_sha256": _sha(old_root),
        "threadblock_proto_set_sha256": expected_threadblock_set,
    }
    assert entry["input_digests"]["classifier_manifest_sha256"] != _sha(
        new_manifest
    )
    assert entry["input_digests"]["dynamic_trace_pb_sha256"] != _sha(
        new_root
    )
    assert read_counts == {
        "manifest.json": 1,
        "dynamic_trace.pb": 1,
        "threadblocks/device_0/stream_0/kernel_1/block_0.pb": 1,
    }


def test_warp_pool_trace_race_fails_before_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trace_pb_path, threadblock_path = _install_warp_pool_cell(
        tmp_path,
        monkeypatch,
        status="matched",
        capabilities=(1, 2, 3, 6, 8),
        include_threadblock=True,
    )
    original_read_snapshot = (
        warp_pool.immutable.PinnedDirectoryReader.read_snapshot
    )
    swapped = False

    def swap_threadblock_after_root(self, relative_path: str, **kwargs):
        nonlocal swapped
        snapshot = original_read_snapshot(self, relative_path, **kwargs)
        if relative_path == "dynamic_trace.pb" and not swapped:
            replacement = threadblock_path.with_name("replacement.pb")
            replacement.write_bytes(b"replacement-generation")
            os.replace(replacement, threadblock_path)
            swapped = True
        return snapshot

    monkeypatch.setattr(
        warp_pool.immutable.PinnedDirectoryReader,
        "read_snapshot",
        swap_threadblock_after_root,
    )
    monkeypatch.setattr(
        warp_pool.lph,
        "_find_kernel_record",
        lambda *_args, **_kwargs: pytest.fail(
            "prediction must not run after an in-snapshot mutation"
        ),
    )

    with pytest.raises(
        warp_pool.immutable.ImmutableArtifactError,
        match="(generation changed|mutated|changed during use)",
    ):
        warp_pool.compute_for_cell(
            slug="cell",
            shape_point="shape",
            cell_row=_identity_row(),
        )

    assert swapped


def test_warp_pool_resource_failure_is_typed_and_closes_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_warp_pool_cell(
        tmp_path,
        monkeypatch,
        status="matched",
        capabilities=(1, 2, 3, 6, 8),
        include_threadblock=True,
    )
    original_open_relative = warp_pool.immutable._open_relative_regular_at

    def exhaust_on_trace_root(root_descriptor: int, relative_path: str):
        if relative_path == "dynamic_trace.pb":
            raise OSError(errno.EMFILE, "fixture descriptor exhaustion")
        return original_open_relative(root_descriptor, relative_path)

    monkeypatch.setattr(
        warp_pool.immutable,
        "_open_relative_regular_at",
        exhaust_on_trace_root,
    )

    closed_readers: list[object] = []
    original_close = warp_pool.immutable.PinnedDirectoryReader.close

    def track_close(self) -> None:
        closed_readers.append(self)
        original_close(self)

    monkeypatch.setattr(
        warp_pool.immutable.PinnedDirectoryReader,
        "close",
        track_close,
    )
    monkeypatch.setattr(
        warp_pool.lph,
        "_find_kernel_record",
        lambda *_args, **_kwargs: pytest.fail(
            "prediction must not run after resource exhaustion"
        ),
    )

    with pytest.raises(
        warp_pool.immutable.ImmutableArtifactResourceError,
        match=warp_pool.immutable.VERIFY_RESOURCE_BOUND,
    ):
        warp_pool.compute_for_cell(
            slug="cell",
            shape_point="shape",
            cell_row=_identity_row(),
        )

    assert len(closed_readers) == 2
    assert closed_readers[0] is not closed_readers[1]
    assert all(reader._binding is None for reader in closed_readers)
