from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.local_labels import capture_budget_route as route
from tools.trace_sampling.counting_pass import BlockStat, CellStats


def _stats(
    n_blocks: int = 60,
    *,
    kernel: str = "candidate_kernel",
    serialized_bytes: int = 0,
    serialized_bytes_exact: bool = False,
) -> CellStats:
    blocks = []
    for block_id in range(n_blocks):
        block = BlockStat(
            block_id=block_id,
            sm_id=0,
            records=100,
            sector_accesses=10,
            serialized_bytes=serialized_bytes,
            warps_per_block=1,
            max_warp_slot=0,
        )
        block.wave = block_id
        blocks.append(block)
    return CellStats(
        cell_id="cell", kernel_name=kernel,
        grid=(n_blocks, 1, 1), block=(32, 1, 1),
        n_blocks=n_blocks, total_records=n_blocks * 100,
        op_class_totals={1: n_blocks * 100}, blocks=blocks,
        per_sm_resident={0: 1}, per_sm_block_ids={0: list(range(n_blocks))},
        n_waves=n_blocks, atomic_records=0,
        atomic_top256_sector_share=0.0,
        per_wave_record_totals={block_id: 100 for block_id in range(n_blocks)},
        per_sm_block_counts={0: n_blocks},
        stats_pass_version=2,
        serialized_bytes_exact=serialized_bytes_exact,
    )


def _deterministic_bytes(label: str, size: int) -> bytes:
    return hashlib.shake_256(label.encode()).digest(size)


def _write_raw_trace(root: Path, stats: CellStats, *, block_size: int,
                     compressible: bool) -> Path:
    root.mkdir(parents=True)
    (root / "dynamic_trace.pb").write_bytes(b"root-envelope-v1")
    (root / "stats.csv").write_text(
        "kernel_id,total_insts,total_reported_insts\n"
        f"1,{stats.total_records},{stats.total_records}\n")
    kernel_dir = root / "threadblocks/device_0/stream_0/kernel_1"
    kernel_dir.mkdir(parents=True)
    for block_id in range(stats.n_blocks):
        payload = (b"x" * block_size if compressible else
                   _deterministic_bytes(f"full-{root.name}-{block_id}", block_size))
        (kernel_dir / f"block_{block_id:04d}.pb").write_bytes(payload)
    return root


def _identity() -> dict:
    return {"roster_source": "sol_execbench", "slug": "cell"}


def _budget(max_bytes: int) -> dict:
    return {
        "max_trace_bytes": max_bytes,
        "max_wall_time_s": 600,
        "sampling_policy": "off",
    }


def _base_protocol() -> dict:
    return {
        "protocol_version": "v2.1",
        "source_commit": "1" * 40,
        "authority_cell_sha256": "2" * 64,
        "capture_matrix_sha256": "3" * 64,
        "preflight_sha256": "4" * 64,
        "pins_sha256": "5" * 64,
        "trace_budget_sha256": "6" * 64,
        "capture_manifest_sha256": "0" * 64,
        "binary_sha256": "7" * 64,
        "tracer_so_sha256": "8" * 64,
        "args_hash": "9" * 64,
    }


def _coverage(stats: CellStats, trace_dir: Path) -> dict:
    return {
        "status": route.preflight.STATUS_OK,
        "detail": "exact full grid",
        "grid_threadblock_count": stats.n_blocks,
        "decoded_threadblock_count": stats.n_blocks,
        "threadblocks_pb_count": len(list((trace_dir / "threadblocks").rglob("*.pb"))),
        "decoded_record_count": stats.total_records,
        "zero_record_threadblock_count": 0,
        "stats_row_count": 1,
        "sampling_sidecar_present": False,
    }


def _write_capture_manifest(
    path: Path,
    trace_dir: Path,
    stats: CellStats,
    budget: dict,
    protocol: dict,
) -> Path:
    _, tree_bytes, _ = route._tree_inventory(trace_dir)
    coverage = _coverage(stats, trace_dir)
    manifest = {
        "schema_version": 1,
        "identity": _identity(),
        "binary_sha256": protocol["binary_sha256"],
        "tracer_so_sha256": protocol["tracer_so_sha256"],
        "max_trace_bytes": budget["max_trace_bytes"],
        "trace_tree_bytes": tree_bytes,
        "launch_count": 1,
        "threadblocks_pb_count": stats.n_blocks,
        "grid_threadblock_count": coverage["grid_threadblock_count"],
        "decoded_threadblock_count": coverage["decoded_threadblock_count"],
        "decoded_record_count": coverage["decoded_record_count"],
        "zero_record_threadblock_count": 0,
        "stats_row_count": 1,
        "expected_records": stats.total_records,
        "admitted_records": stats.total_records,
        "drop_rate": 0.0,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _patch_full_validation(monkeypatch, stats_by_root: dict[Path, CellStats]) -> None:
    def stats_for(path: Path) -> CellStats:
        resolved = Path(path).resolve()
        for root, stats in stats_by_root.items():
            if resolved == root.resolve():
                return stats
        raise AssertionError(f"unexpected trace root {resolved}")

    monkeypatch.setattr(
        route, "validate_trace_dir", lambda _path: SimpleNamespace(ok=True))
    monkeypatch.setattr(
        route, "analyze_trace_dir", lambda path, _cell: stats_for(path))

    def root_validator(pb_path, _repo):
        stats = stats_for(Path(pb_path).parent)
        return {
            "status": route.preflight.STATUS_OK,
            "detail": "ok",
            "launch_count": 1,
            "trace_contract_id": "sm86_addr_carry_v1",
            "memory_payload_classes": [1, 2, 3, 6, 8],
            "launches": [{
                "device_id": 0, "stream_id": 0, "kernel_id": 1,
                "kernel_name": stats.kernel_name,
                "grid_dim": list(stats.grid), "block_dim": list(stats.block),
            }],
        }

    def stats_details(stats_path):
        stats = stats_for(Path(stats_path).parent)
        return {
            "expected": stats.total_records,
            "admitted": stats.total_records,
            "row_count": 1,
            "error": None,
        }

    def full_coverage(trace_dir, _root, _expected, _admitted, _rows):
        return _coverage(stats_for(trace_dir), Path(trace_dir))

    monkeypatch.setattr(route.preflight, "_validate_root_trace", root_validator)
    monkeypatch.setattr(route.preflight, "_parse_stats_admission_details", stats_details)
    monkeypatch.setattr(route.preflight, "_analyze_full_trace_coverage", full_coverage)


def _protocol_with_manifest(protocol: dict, manifest: Path) -> dict:
    return {**protocol, "capture_manifest_sha256": route.sha256_file(manifest)}


def _config(
    max_bytes: int,
    *,
    policy: str = "typed_unavailable",
    variant: str | None = None,
) -> route.RouteConfig:
    return route.RouteConfig(
        selection_max_raw_bytes=max(1, max_bytes // 2),
        staging_limit_bytes=4_000_000,
        gate_forbidden_policy=policy,
        reduced_shape_variant=variant,
    )


def _never(_request):
    raise AssertionError("capture callback must not run")


def _receipt(
    request: route.CaptureRequest,
    *,
    manifest: Path | None = None,
    matched_label_evidence: Path | None = None,
):
    return route.CaptureReceipt(
        trace_dir=request.output_dir,
        request_sha256=request.request_sha256,
        binary_sha256="7" * 64,
        tracer_so_sha256="8" * 64,
        capture_manifest=manifest,
        matched_label_evidence=matched_label_evidence,
    )


def test_lossless_full_route_is_replayable_and_never_runs_callbacks(tmp_path, monkeypatch):
    stats = _stats()
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=4000, compressible=True)
    budget = _budget(80_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})

    output = tmp_path / "retained"
    result = route.route_trace_budget(
        full, manifest, output, cell_id="cell", identity=_identity(),
        budget=budget, protocol=protocol, config=_config(budget["max_trace_bytes"]),
        callbacks=route.RouteCallbacks(_never, _never),
    )

    assert result["method"] == "lossless_compressed_full"
    assert result["outcome"] == "admitted"
    assert result["retained"]["bytes"] <= budget["max_trace_bytes"]
    assert result["result_sha256"] == route.canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"})
    verified = route.verify_replay_bundle(output)
    assert verified["payload_sha256"] == json.loads(
        (output / "trace_sampling.json").read_text()
    )["hashes"]["decompressed_payload_sha256"]

    materialized = tmp_path / "materialized"
    replay = route.materialize_replay_bundle(output, materialized)
    assert replay["source_tree_sha256"] == result["full_trace"]["tree_sha256"]
    assert (materialized / "trace_sampling.json").is_file()
    source_rows = [
        row for row in route._tree_inventory(materialized)[0]
        if row["relative_path"] != "trace_sampling.json"
    ]
    assert route.canonical_sha256(source_rows) == result["full_trace"]["tree_sha256"]


def test_route_refuses_stale_full_trace_producer_with_typed_reason(
        tmp_path, monkeypatch):
    stats = _stats()
    stats.stats_pass_version = 1
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=4000, compressible=True)
    budget = _budget(80_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})

    with pytest.raises(route.BudgetRouteRefusal) as caught:
        route.route_trace_budget(
            full, manifest, tmp_path / "retained", cell_id="cell",
            identity=_identity(), budget=budget, protocol=protocol,
            config=_config(budget["max_trace_bytes"]),
            callbacks=route.RouteCallbacks(_never, _never),
        )
    assert caught.value.reason == "trace_sampling_producer_unsupported"
    assert "stats-pass v1" in caught.value.detail


def test_lossless_bundle_hashes_payloads_in_canonical_relative_path_order(
        tmp_path, monkeypatch):
    stats = _stats(2)
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=1, compressible=True)
    original = sorted((full / "threadblocks").rglob("*.pb"))
    first = b"first" * 20_000
    second = b"second" * 20_000
    first_path = full / "threadblocks/a.pb"
    second_path = full / "threadblocks/a/b.pb"
    first_path.parent.mkdir(exist_ok=True)
    second_path.parent.mkdir(exist_ok=True)
    original[0].rename(first_path)
    original[1].rename(second_path)
    first_path.write_bytes(first)
    second_path.write_bytes(second)
    budget = _budget(40_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    output = tmp_path / "retained"

    route.route_trace_budget(
        full, manifest, output, cell_id="cell", identity=_identity(),
        budget=budget, protocol=protocol, config=_config(budget["max_trace_bytes"]),
        callbacks=route.RouteCallbacks(_never, _never),
    )

    sidecar = json.loads((output / "trace_sampling.json").read_text())
    assert sidecar["hashes"]["decompressed_payload_sha256"] == hashlib.sha256(
        first + second).hexdigest()
    assert route.verify_replay_bundle(output)["status"] == "ok"


def test_sidecar_construction_failure_is_typed_and_removes_candidate(
        tmp_path, monkeypatch):
    stats = _stats(2)
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=4000, compressible=True)
    budget = _budget(100)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})

    def invalid_sidecar(**_kwargs):
        raise ValueError("synthetic sidecar construction failure")

    monkeypatch.setattr(route, "build_sidecar", invalid_sidecar)
    output = tmp_path / "retained"
    candidate = tmp_path / ".retained.ac9_work/lossless_full_candidate"
    with pytest.raises(route.BudgetRouteRefusal) as caught:
        route.route_trace_budget(
            full, manifest, output, cell_id="cell", identity=_identity(),
            budget=budget, protocol=protocol, config=_config(budget["max_trace_bytes"]),
            callbacks=route.RouteCallbacks(_never, _never),
        )
    assert caught.value.reason == "trace_sampling_sidecar_invalid"
    assert "synthetic sidecar construction failure" in caught.value.detail
    assert not candidate.exists()


@pytest.mark.parametrize(
    ("exception_type", "message"),
    [
        (RuntimeError, "zstd compress failed"),
        (OSError, "codec output write failed"),
    ],
)
def test_compression_layer_failure_is_typed_and_removes_partial_candidate(
        tmp_path, monkeypatch, exception_type, message):
    stats = _stats(2)
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=1000, compressible=False,
    )
    budget = _budget(100)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol,
    )
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    original_inventory = route._tree_inventory(full)

    def failed_compression(_source, destination):
        partial = (
            destination
            / "threadblocks/device_0/stream_0/kernel_1/block_0000.pb.zst"
        )
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"partial-not-admissible")
        raise exception_type(message)

    monkeypatch.setattr(route, "compress_trace_dir", failed_compression)
    output = tmp_path / "retained"
    work_root = tmp_path / ".retained.ac9_work"
    candidate = work_root / "lossless_full_candidate"
    with pytest.raises(route.BudgetRouteRefusal) as caught:
        route.route_trace_budget(
            full,
            manifest,
            output,
            cell_id="cell",
            identity=_identity(),
            budget=budget,
            protocol=protocol,
            config=_config(budget["max_trace_bytes"]),
            callbacks=route.RouteCallbacks(_never, _never),
        )
    assert caught.value.reason == "lossless_compression_failed"
    assert caught.value.detail == f"{exception_type.__name__}: {message}"
    assert isinstance(caught.value.__cause__, exception_type)
    assert not output.exists()
    assert not candidate.exists()
    assert route._tree_inventory(full) == original_inventory


def test_typed_compression_refusal_is_not_masked_and_candidate_is_removed(
        tmp_path, monkeypatch):
    stats = _stats(2)
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=1000, compressible=False,
    )
    budget = _budget(100)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol,
    )
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})

    def typed_refusal(_source, destination):
        (destination / "partial.pb.zst").write_bytes(b"partial")
        raise route.BudgetRouteRefusal(
            "trace_sampling_sidecar_invalid", "synthetic typed refusal",
        )

    monkeypatch.setattr(route, "compress_trace_dir", typed_refusal)
    output = tmp_path / "retained"
    candidate = tmp_path / ".retained.ac9_work/lossless_full_candidate"
    with pytest.raises(route.BudgetRouteRefusal) as caught:
        route.route_trace_budget(
            full,
            manifest,
            output,
            cell_id="cell",
            identity=_identity(),
            budget=budget,
            protocol=protocol,
            config=_config(budget["max_trace_bytes"]),
            callbacks=route.RouteCallbacks(_never, _never),
        )
    assert caught.value.reason == "trace_sampling_sidecar_invalid"
    assert caught.value.detail == "synthetic typed refusal"
    assert not output.exists()
    assert not candidate.exists()


def _install_sampling_callbacks(monkeypatch, full_stats: CellStats, *, stale=False):
    counting_stats = _stats(full_stats.n_blocks, serialized_bytes=0)
    pass2_stats = _stats(
        full_stats.n_blocks, serialized_bytes=128, serialized_bytes_exact=True)
    selected_by_path: dict[Path, list[int]] = {}

    def counting(request: route.CaptureRequest):
        assert request.kind == "trace_sampling_counting_v2"
        request.output_dir.mkdir(parents=True)
        (request.output_dir / "dynamic_trace.pb").write_bytes(b"count-root")
        (request.output_dir / "stats.csv").write_text("counting\n")
        extra = request.output_dir / "extra_info"
        extra.mkdir()
        (extra / "trace_sampling_capture.json").write_text('{"mode": 1}\n')
        receipt = _receipt(request)
        if stale:
            receipt = route.CaptureReceipt(
                trace_dir=receipt.trace_dir, request_sha256="0" * 64,
                binary_sha256=receipt.binary_sha256,
                tracer_so_sha256=receipt.tracer_so_sha256,
            )
        return receipt

    def selective(request: route.CaptureRequest):
        assert request.kind == "trace_sampling_selective_v2"
        request.output_dir.mkdir(parents=True)
        (request.output_dir / "dynamic_trace.pb").write_bytes(b"select-root")
        (request.output_dir / "stats.csv").write_text("selective\n")
        extra = request.output_dir / "extra_info"
        extra.mkdir()
        (extra / "trace_sampling_capture.json").write_text(
            json.dumps({
                "mode": 2,
                "selected_block_ids": list(request.selected_block_ids),
            }) + "\n")
        kernel = request.output_dir / "threadblocks/device_0/stream_0/kernel_1"
        kernel.mkdir(parents=True)
        selected_by_path[request.output_dir] = list(request.selected_block_ids)
        for block_id in request.selected_block_ids:
            (kernel / f"block_{block_id:04d}.pb").write_bytes(
                _deterministic_bytes(f"pass2-{block_id}", 128))
        return _receipt(request)

    def load_stats(path, _cell):
        return counting_stats if Path(path).name == "counting" else pass2_stats

    def verify_selection(path, selected):
        assert selected_by_path[Path(path)] == list(selected)
        raw = (Path(path) / "extra_info/trace_sampling_capture.json").read_bytes()
        return hashlib.sha256(raw).hexdigest()

    def verify_payload(path, _grid, selected, serialized):
        assert selected_by_path[Path(path)] == list(selected)
        assert serialized == {block_id: 128 for block_id in selected}
        return (
            {block_id: 100 for block_id in selected},
            {block_id: 128 for block_id in selected},
        )

    def validate_sampling_root(path, _expected_stats, *, mode):
        capture = json.loads(
            (Path(path) / "extra_info/trace_sampling_capture.json").read_text())
        assert capture["mode"] == mode
        if mode == 1:
            assert not list((Path(path) / "threadblocks").rglob("*.pb")) \
                if (Path(path) / "threadblocks").exists() else True

    monkeypatch.setattr(route, "cell_stats_from_capture", load_stats)
    monkeypatch.setattr(route, "verify_capture_selection", verify_selection)
    monkeypatch.setattr(route, "verify_pass2_payload", verify_payload)
    monkeypatch.setattr(route, "_validate_sampling_root", validate_sampling_root)
    return route.RouteCallbacks(counting, selective), counting_stats, pass2_stats


def test_over_limit_full_routes_to_verified_schema_v2_subset(tmp_path, monkeypatch):
    stats = _stats()
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=3000, compressible=False)
    budget = _budget(65_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    callbacks, _, _ = _install_sampling_callbacks(monkeypatch, stats)

    output = tmp_path / "retained"
    result = route.route_trace_budget(
        full, manifest, output, cell_id="cell", identity=_identity(),
        budget=budget, protocol=protocol, config=_config(budget["max_trace_bytes"]),
        callbacks=callbacks,
    )

    assert result["method"] == "wave_segment_stratified_subset"
    assert result["trace_sampling_schema_version"] == 2
    sidecar = json.loads((output / "trace_sampling.json").read_text())
    assert sidecar["selection_reconciliation"]["status"] == "fixed_point"
    payload_manifest = sidecar["hashes"]["per_block_raw_sha256_manifest"]
    assert payload_manifest.startswith("trace-payload-file-manifest-v1\n")
    assert payload_manifest.endswith("\n")
    envelope = json.loads((output / "ac9_replay_envelope.json").read_text())
    assert len(envelope["per_block_raw_sha256_manifest"]) == 64
    assert envelope["per_block_raw_sha256_manifest"] != payload_manifest
    assert route.verify_replay_bundle(output)["status"] == "ok"
    assert result["retained"]["bytes"] <= budget["max_trace_bytes"]


def test_route_refuses_stale_selective_producer_with_typed_reason(
        tmp_path, monkeypatch):
    stats = _stats()
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=3000, compressible=False)
    budget = _budget(65_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    callbacks, _, pass2_stats = _install_sampling_callbacks(monkeypatch, stats)
    pass2_stats.stats_pass_version = 1

    with pytest.raises(route.BudgetRouteRefusal) as caught:
        route.route_trace_budget(
            full, manifest, tmp_path / "retained", cell_id="cell",
            identity=_identity(), budget=budget, protocol=protocol,
            config=_config(budget["max_trace_bytes"]), callbacks=callbacks,
        )
    assert caught.value.reason == "trace_sampling_producer_unsupported"
    assert "selective capture" in caught.value.detail


def test_replay_verifier_rejects_tampered_compressed_payload(tmp_path, monkeypatch):
    stats = _stats()
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=4000, compressible=True)
    budget = _budget(80_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    output = tmp_path / "retained"
    route.route_trace_budget(
        full, manifest, output, cell_id="cell", identity=_identity(),
        budget=budget, protocol=protocol, config=_config(budget["max_trace_bytes"]),
        callbacks=route.RouteCallbacks(_never, _never),
    )
    target = next(output.rglob("*.pb.zst"))
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 0x01
    target.write_bytes(raw)
    with pytest.raises(route.BudgetRouteRefusal, match="replay_retained_hash_mismatch"):
        route.verify_replay_bundle(output)


def _reseal_envelope(output: Path, envelope: dict) -> None:
    envelope["entries_sha256"] = route.canonical_sha256(envelope["entries"])
    envelope.pop("envelope_sha256", None)
    envelope["envelope_sha256"] = route.canonical_sha256(envelope)
    (output / "ac9_replay_envelope.json").write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n")


def _lossless_bundle(tmp_path, monkeypatch) -> Path:
    stats = _stats()
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=4000, compressible=True)
    budget = _budget(80_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    output = tmp_path / "retained"
    route.route_trace_budget(
        full, manifest, output, cell_id="cell", identity=_identity(),
        budget=budget, protocol=protocol, config=_config(budget["max_trace_bytes"]),
        callbacks=route.RouteCallbacks(_never, _never),
    )
    return output


def test_replay_verifier_rejects_freshly_resealed_entry_reordering(
        tmp_path, monkeypatch):
    output = _lossless_bundle(tmp_path, monkeypatch)
    envelope = json.loads((output / "ac9_replay_envelope.json").read_text())
    envelope["entries"][0], envelope["entries"][1] = (
        envelope["entries"][1], envelope["entries"][0])
    _reseal_envelope(output, envelope)

    with pytest.raises(route.BudgetRouteRefusal) as caught:
        route.verify_replay_bundle(output)
    assert caught.value.reason == "replay_envelope_malformed"
    assert "canonically path-ordered" in caught.value.detail


def test_replay_verifier_rejects_external_retained_path_before_reading_it(
        tmp_path, monkeypatch):
    output = _lossless_bundle(tmp_path, monkeypatch)
    envelope = json.loads((output / "ac9_replay_envelope.json").read_text())
    entry = next(
        row for row in envelope["entries"]
        if row["source_relative_path"] == "dynamic_trace.pb")
    outside = output.parent / "outside"
    outside.write_bytes((output / entry["retained_relative_path"]).read_bytes())
    entry["retained_relative_path"] = "../outside"
    envelope["entries"].sort(key=lambda row: row["retained_relative_path"])
    _reseal_envelope(output, envelope)
    original_read_bytes = Path.read_bytes

    def guarded_read(path):
        if path.resolve(strict=False) == outside.resolve(strict=False):
            raise AssertionError("verifier read outside its bundle")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(route.BudgetRouteRefusal) as caught:
        route.verify_replay_bundle(output)
    assert caught.value.reason == "replay_envelope_malformed"


def test_replay_verifier_rejects_resealed_identity_encoded_pb_payload(
        tmp_path, monkeypatch):
    output = _lossless_bundle(tmp_path, monkeypatch)
    envelope = json.loads((output / "ac9_replay_envelope.json").read_text())
    target = next(row for row in envelope["entries"] if row["encoding"] == "zstd")
    packed_path = output / target["retained_relative_path"]
    raw = route._zstd(packed_path.read_bytes(), decompress=True)
    identity_path = output / target["source_relative_path"]
    identity_path.write_bytes(raw)
    packed_path.unlink()
    target.update({
        "retained_relative_path": target["source_relative_path"],
        "encoding": "identity",
        "retained_size_bytes": len(raw),
        "retained_sha256": hashlib.sha256(raw).hexdigest(),
    })

    payload_entries = [
        row for row in envelope["entries"] if row["encoding"] == "zstd"]
    payload_rows = [{
        "relative_path": row["source_relative_path"],
        "size_bytes": row["raw_size_bytes"],
        "sha256": row["raw_sha256"],
    } for row in payload_entries]
    payload_chunks = []
    for row in sorted(payload_entries, key=lambda item: item["source_relative_path"]):
        payload_chunks.append(route._zstd(
            (output / row["retained_relative_path"]).read_bytes(), decompress=True))
    payload_sha = hashlib.sha256(b"".join(payload_chunks)).hexdigest()

    sidecar_path = output / "trace_sampling.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["bytes"]["raw"] = sum(row["raw_size_bytes"] for row in payload_entries)
    sidecar["bytes"]["compressed"] = sum(
        row["retained_size_bytes"] for row in payload_entries)
    sidecar["hashes"]["decompressed_payload_sha256"] = payload_sha
    sidecar["hashes"]["per_block_raw_sha256_manifest"] = (
        route._canonical_payload_file_manifest(payload_rows))
    sidecar["hashes"].pop("sidecar_sha256")
    sidecar["hashes"]["sidecar_sha256"] = route.sidecar_sha256(sidecar)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")

    generated = next(
        row for row in envelope["entries"] if row["encoding"] == "generated")
    generated["retained_size_bytes"] = sidecar_path.stat().st_size
    generated["retained_sha256"] = route.sha256_file(sidecar_path)
    envelope["threadblock_count"] = len(payload_entries)
    envelope["decompressed_threadblocks_sha256"] = payload_sha
    envelope["per_block_raw_sha256_manifest"] = route.canonical_sha256(payload_rows)
    envelope["entries"].sort(key=lambda row: row["retained_relative_path"])
    _reseal_envelope(output, envelope)

    with pytest.raises(route.BudgetRouteRefusal) as caught:
        route.verify_replay_bundle(output)
    assert caught.value.reason == "replay_envelope_malformed"
    assert "requires zstd encoding" in caught.value.detail


def test_replay_verifier_rejects_pb_named_source_directory(
        tmp_path, monkeypatch):
    output = _lossless_bundle(tmp_path, monkeypatch)
    envelope = json.loads((output / "ac9_replay_envelope.json").read_text())
    entry = next(
        row for row in envelope["entries"]
        if row["source_relative_path"] == "stats.csv")
    old_path = output / entry["retained_relative_path"]
    source_rel = "threadblocks/dir.pb/aux.txt"
    new_path = output / source_rel
    new_path.parent.mkdir()
    old_path.rename(new_path)
    entry["source_relative_path"] = source_rel
    entry["retained_relative_path"] = source_rel
    raw_rows = [{
        "relative_path": row["source_relative_path"],
        "size_bytes": row["raw_size_bytes"],
        "sha256": row["raw_sha256"],
    } for row in envelope["entries"] if row["encoding"] != "generated"]
    raw_rows.sort(key=lambda row: row["relative_path"])
    envelope["source_tree_sha256"] = route.canonical_sha256(raw_rows)
    envelope["entries"].sort(key=lambda row: row["retained_relative_path"])
    _reseal_envelope(output, envelope)

    with pytest.raises(route.BudgetRouteRefusal) as caught:
        route.verify_replay_bundle(output)
    assert caught.value.reason == "replay_envelope_malformed"
    assert ".pb-named directory" in caught.value.detail


def test_replay_verifier_types_noncanonical_payload_name(
        tmp_path, monkeypatch):
    output = _lossless_bundle(tmp_path, monkeypatch)
    envelope = json.loads((output / "ac9_replay_envelope.json").read_text())
    entry = next(row for row in envelope["entries"] if row["encoding"] == "zstd")
    old_path = output / entry["retained_relative_path"]
    source_rel = "threadblocks/a b.pb"
    new_rel = source_rel + ".zst"
    new_path = output / new_rel
    new_path.parent.mkdir(exist_ok=True)
    old_path.rename(new_path)
    entry["source_relative_path"] = source_rel
    entry["retained_relative_path"] = new_rel
    envelope["entries"].sort(key=lambda row: row["retained_relative_path"])
    _reseal_envelope(output, envelope)

    with pytest.raises(route.BudgetRouteRefusal) as caught:
        route.verify_replay_bundle(output)
    assert caught.value.reason == "replay_envelope_malformed"
    assert "invalid threadblock payload name" in caught.value.detail


def test_gate_forbidden_uses_explicit_typed_unavailable_result(tmp_path, monkeypatch):
    stats = _stats(6)
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=12_000, compressible=False)
    budget = _budget(25_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    callbacks, _, _ = _install_sampling_callbacks(monkeypatch, stats)
    callbacks = route.RouteCallbacks(callbacks.counting_capture, _never)

    output = tmp_path / "retained"
    result = route.route_trace_budget(
        full, manifest, output, cell_id="cell", identity=_identity(),
        budget=budget, protocol=protocol,
        config=_config(budget["max_trace_bytes"], policy="typed_unavailable"),
        callbacks=callbacks,
    )

    assert result["method"] == "unavailable"
    assert result["outcome"] == "typed_unavailable"
    assert result["typed_reason"] == "gate_forced"
    assert json.loads((output / "trace_sampling.json").read_text())["schema_version"] == 2
    evidence = json.loads((output / "ac9_unavailable_evidence.json").read_text())
    assert "wave_gate" in evidence["forbidding_gates"]


def test_pass2_exact_certainty_gate_also_routes_to_typed_unavailable(tmp_path, monkeypatch):
    stats = _stats()
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=3000, compressible=False)
    budget = _budget(65_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    callbacks, _, _ = _install_sampling_callbacks(monkeypatch, stats)
    config = route.RouteConfig(
        selection_max_raw_bytes=1000,
        staging_limit_bytes=4_000_000,
        gate_forbidden_policy="typed_unavailable",
    )

    result = route.route_trace_budget(
        full, manifest, tmp_path / "retained", cell_id="cell",
        identity=_identity(), budget=budget, protocol=protocol,
        config=config, callbacks=callbacks,
    )

    assert result["method"] == "unavailable"
    assert result["outcome"] == "typed_unavailable"
    sidecar = json.loads((tmp_path / "retained/trace_sampling.json").read_text())
    assert sidecar["gates"]["certainty_budget_exceeded"] is True
    assert sidecar["gates"]["verdict"] in {
        "skew_to_reduced_shape", "certainty_budget_to_reduced_shape"}


def test_reconciliation_exhaustion_seals_budget_limited_terminal(
        tmp_path, monkeypatch):
    stats = _stats()
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=3000, compressible=False)
    budget = _budget(65_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    callbacks, _, _ = _install_sampling_callbacks(monkeypatch, stats)
    original_selected = route._selected_block_ids
    calls = 0

    def never_fixed(plan):
        nonlocal calls
        calls += 1
        selected = original_selected(plan)
        if calls == 1:
            return selected
        replacement = next(
            block_id for block_id in range(stats.n_blocks)
            if block_id not in selected
        )
        return sorted([*selected[1:], replacement])

    monkeypatch.setattr(route, "_selected_block_ids", never_fixed)
    config = route.RouteConfig(
        selection_max_raw_bytes=budget["max_trace_bytes"] * 3 // 4,
        staging_limit_bytes=4_000_000,
        gate_forbidden_policy="typed_unavailable",
        max_reconcile_attempts=1,
    )
    output = tmp_path / "retained"
    result = route.route_trace_budget(
        full, manifest, output, cell_id="cell", identity=_identity(),
        budget=budget, protocol=protocol, config=config,
        callbacks=callbacks,
    )

    assert result["outcome"] == "typed_unavailable"
    assert result["method"] == "unavailable"
    assert result["typed_reason"] == "budget_limited_wide_ci"
    sidecar = json.loads((output / "trace_sampling.json").read_text())
    assert sidecar["reason"] == "byte_budget"
    evidence = json.loads(
        (output / "ac9_unavailable_evidence.json").read_text()
    )
    assert evidence["typed_reason"] == "budget_limited_wide_ci"
    assert "placement_reconciliation_exhausted" in evidence["terminal_detail"]
    assert result["result_sha256"] == route.canonical_sha256({
        key: value for key, value in result.items()
        if key != "result_sha256"
    })


def test_gate_forbidden_reduced_shape_requires_explicit_variant_and_receipt(tmp_path):
    with pytest.raises(route.BudgetRouteRefusal, match="explicit certified variant"):
        route._validate_config(
            route.RouteConfig(
                selection_max_raw_bytes=10, staging_limit_bytes=100,
                gate_forbidden_policy="reduced_shape"),
            20,
        )


def test_reduced_shape_policy_produces_matched_full_coverage_bundle(tmp_path, monkeypatch):
    full_stats = _stats(6)
    reduced_stats = _stats(3)
    full = _write_raw_trace(
        tmp_path / "full", full_stats, block_size=12_000, compressible=False)
    budget = _budget(30_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, full_stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    stats_by_root = {full: full_stats}
    _patch_full_validation(monkeypatch, stats_by_root)
    callbacks, _, _ = _install_sampling_callbacks(monkeypatch, full_stats)

    def reduced(request: route.CaptureRequest):
        assert request.reduced_shape_variant == "dataset_shrunk"
        _write_raw_trace(
            request.output_dir, reduced_stats, block_size=256, compressible=True)
        stats_by_root[request.output_dir] = reduced_stats
        label_root = request.output_dir / "matched_label"
        label_root.mkdir()
        label_artifacts = []
        for kind, name in (
            ("label_sidecar", "label.json"),
            ("label_run_manifest", "label_run_manifest.json"),
            ("ncu_report", "report.ncu-rep"),
        ):
            artifact_path = label_root / name
            artifact_path.write_bytes(f"{kind}-reduced\n".encode())
            label_artifacts.append({
                "artifact_kind": kind,
                "relative_path": artifact_path.relative_to(request.output_dir).as_posix(),
                "sha256": route.sha256_file(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
            })
        evidence_path = (
            request.output_dir / "extra_info/matched_reduced_label_evidence.json")
        evidence_path.parent.mkdir(exist_ok=True)
        evidence_path.write_text(json.dumps({
            "schema_version": 1,
            "kind": "matched_reduced_shape_label_evidence",
            "identity": _identity(),
            "reduced_shape_variant": "dataset_shrunk",
            "args_hash": protocol["args_hash"],
            "binary_sha256": protocol["binary_sha256"],
            "artifacts": label_artifacts,
        }, indent=2, sort_keys=True) + "\n")
        reduced_manifest = _write_capture_manifest(
            tmp_path / "reduced_capture.json", request.output_dir,
            reduced_stats, budget, protocol)
        return _receipt(
            request, manifest=reduced_manifest,
            matched_label_evidence=evidence_path)

    output = tmp_path / "retained"
    result = route.route_trace_budget(
        full, manifest, output, cell_id="cell", identity=_identity(),
        budget=budget, protocol=protocol,
        config=_config(
            budget["max_trace_bytes"], policy="reduced_shape",
            variant="dataset_shrunk"),
        callbacks=route.RouteCallbacks(
            callbacks.counting_capture, _never, reduced),
    )

    assert result["method"] == "reduced_shape_matched_label"
    sidecar = json.loads((output / "trace_sampling.json").read_text())
    assert sidecar["label_pairing"] == "matched_reduced_shape_label"
    assert sidecar["shape"]["original_grid"] == {"x": 6, "y": 1, "z": 1}
    assert sidecar["shape"]["captured_grid"] == {"x": 3, "y": 1, "z": 1}
    assert route.verify_replay_bundle(output)["status"] == "ok"
    kinds = {artifact["artifact_kind"] for artifact in result["retained"]["artifacts"]}
    assert {"label_sidecar", "label_run_manifest", "ncu_report"}.issubset(kinds)


def test_empty_full_trace_refuses_vacuous_coverage_before_callbacks(tmp_path):
    full = tmp_path / "empty"
    (full / "threadblocks").mkdir(parents=True)
    (full / "dynamic_trace.pb").write_bytes(b"x")
    (full / "stats.csv").write_text("x\n")
    protocol = _base_protocol()
    manifest = tmp_path / "capture.json"
    manifest.write_text("{}\n")
    protocol = _protocol_with_manifest(protocol, manifest)
    with pytest.raises(route.BudgetRouteRefusal, match="trace_payload_empty"):
        route.route_trace_budget(
            full, manifest, tmp_path / "out", cell_id="cell",
            identity=_identity(), budget=_budget(1), protocol=protocol,
            config=_config(1), callbacks=route.RouteCallbacks(_never, _never),
        )


def test_full_trace_with_producer_drop_is_typed_truncated(tmp_path, monkeypatch):
    stats = _stats(2)
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=1000, compressible=False)
    budget = _budget(100)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    monkeypatch.setattr(
        route.preflight, "_parse_stats_admission_details",
        lambda _path: {
            "expected": stats.total_records,
            "admitted": stats.total_records - 1,
            "row_count": 1,
            "error": None,
        },
    )
    with pytest.raises(route.BudgetRouteRefusal, match="trace_truncated_or_dropped"):
        route.route_trace_budget(
            full, manifest, tmp_path / "out", cell_id="cell",
            identity=_identity(), budget=budget, protocol=protocol,
            config=_config(budget["max_trace_bytes"]),
            callbacks=route.RouteCallbacks(_never, _never),
        )


def test_purported_full_trace_with_sampling_sidecar_is_refused(tmp_path):
    stats = _stats(2)
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=1000, compressible=False)
    (full / "trace_sampling.json").write_text(
        '{"method":"deterministic_every_k"}\n')
    manifest = tmp_path / "capture.json"
    manifest.write_text("{}\n")
    protocol = _protocol_with_manifest(_base_protocol(), manifest)
    with pytest.raises(route.BudgetRouteRefusal, match="legacy_stride_evidence_refused"):
        route.route_trace_budget(
            full, manifest, tmp_path / "out", cell_id="cell",
            identity=_identity(), budget=_budget(100), protocol=protocol,
            config=_config(100), callbacks=route.RouteCallbacks(_never, _never),
        )


def test_stale_manifest_identity_and_hash_are_refused(tmp_path, monkeypatch):
    stats = _stats(2)
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=1000, compressible=False)
    budget = _budget(100)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    payload = json.loads(manifest.read_text())
    payload["identity"]["slug"] = "stale-cell"
    manifest.write_text(json.dumps(payload) + "\n")
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    with pytest.raises(route.BudgetRouteRefusal, match="capture_manifest_stale"):
        route.route_trace_budget(
            full, manifest, tmp_path / "out", cell_id="cell",
            identity=_identity(), budget=budget, protocol=protocol,
            config=_config(budget["max_trace_bytes"]),
            callbacks=route.RouteCallbacks(_never, _never),
        )


def test_legacy_stride_budget_is_refused_before_trace_consumption(tmp_path):
    budget = _budget(100)
    budget["sampling_policy"] = "deterministic_every_k"
    with pytest.raises(route.BudgetRouteRefusal, match="legacy_stride_evidence_refused"):
        route.route_trace_budget(
            tmp_path / "missing", tmp_path / "missing.json", tmp_path / "out",
            cell_id="cell", identity=_identity(), budget=budget,
            protocol=_base_protocol(), config=_config(100),
            callbacks=route.RouteCallbacks(_never, _never),
        )


def test_stale_counting_receipt_is_refused(tmp_path, monkeypatch):
    stats = _stats()
    full = _write_raw_trace(
        tmp_path / "full", stats, block_size=3000, compressible=False)
    budget = _budget(65_000)
    protocol = _base_protocol()
    manifest = _write_capture_manifest(
        tmp_path / "capture.json", full, stats, budget, protocol)
    protocol = _protocol_with_manifest(protocol, manifest)
    _patch_full_validation(monkeypatch, {full: stats})
    callbacks, _, _ = _install_sampling_callbacks(monkeypatch, stats, stale=True)
    with pytest.raises(route.BudgetRouteRefusal, match="capture_receipt_stale"):
        route.route_trace_budget(
            full, manifest, tmp_path / "out", cell_id="cell",
            identity=_identity(), budget=budget, protocol=protocol,
            config=_config(budget["max_trace_bytes"]), callbacks=callbacks,
        )


def test_capacity_margin_ledger_and_fail_closed_boundaries():
    ledger = route.validate_capacity_margins(
        declared_retained_bytes=[10, 20, 30], existing_retained_bytes=5,
        retention_capacity_bytes=100, retention_reserve_bytes=25,
        per_cell_staging_bytes=[40, 30, 20], existing_staging_bytes=5,
        staging_capacity_bytes=100, staging_reserve_bytes=20,
        max_concurrent_captures=1,
    )
    assert ledger["retention_headroom_after_reserve_bytes"] == 10
    assert ledger["staging_headroom_after_reserve_bytes"] == 35
    unsigned = dict(ledger)
    unsigned.pop("ledger_sha256")
    assert ledger["ledger_sha256"] == route.canonical_sha256(unsigned)

    with pytest.raises(route.BudgetRouteRefusal, match="aggregate_retention_margin_exceeded"):
        route.validate_capacity_margins(
            declared_retained_bytes=[71], existing_retained_bytes=5,
            retention_capacity_bytes=100, retention_reserve_bytes=25,
            per_cell_staging_bytes=[1], existing_staging_bytes=0,
            staging_capacity_bytes=10, staging_reserve_bytes=1,
        )
    with pytest.raises(route.BudgetRouteRefusal, match="aggregate_staging_margin_exceeded"):
        route.validate_capacity_margins(
            declared_retained_bytes=[1], existing_retained_bytes=0,
            retention_capacity_bytes=10, retention_reserve_bytes=1,
            per_cell_staging_bytes=[50, 40], existing_staging_bytes=5,
            staging_capacity_bytes=100, staging_reserve_bytes=10,
            max_concurrent_captures=2,
        )
