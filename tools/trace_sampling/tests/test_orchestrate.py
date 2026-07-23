"""Orchestration: seeded segment selection + a live two-pass integration."""
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from admissibility_gate import MIN_SEGMENTS
from counting_pass import BlockStat, CellStats
from orchestrate import (DEFAULT_TRACER_SO, SamplingRefusal, orchestrate_cell,
                         select_segments)
from selection import SelectionPlanError
from sidecar import sidecar_sha256, validate_sidecar

_REPO = Path(__file__).resolve().parents[3]


def _have_live():
    return (DEFAULT_TRACER_SO.is_file()
            and shutil.which("nvidia-smi") is not None
            and Path("/usr/local/cuda-13.2/bin/nvcc").is_file())


def _value(source, block_id, wave, sm, seq):
    return source(block_id, wave, sm, seq) if callable(source) else source


def _stats_for_placement(per_sm_block_ids, per_sm_resident, *, records=100,
                         sectors=10, serialized_bytes=10,
                         kernel_name="k", serialized_bytes_exact=True):
    blocks = []
    per_wave_record_totals = {}
    total_records = 0
    for sm, block_ids in sorted(per_sm_block_ids.items()):
        resident = per_sm_resident[sm]
        for seq, block_id in enumerate(block_ids):
            wave = seq // resident
            block_records = _value(records, block_id, wave, sm, seq)
            block = BlockStat(
                block_id=block_id, sm_id=sm, records=block_records,
                sector_accesses=_value(sectors, block_id, wave, sm, seq),
                serialized_bytes=_value(
                    serialized_bytes, block_id, wave, sm, seq),
            )
            block.wave = wave
            blocks.append(block)
            total_records += block_records
            per_wave_record_totals[wave] = (
                per_wave_record_totals.get(wave, 0) + block_records)
    blocks.sort(key=lambda block: block.block_id)
    assert [block.block_id for block in blocks] == list(range(len(blocks)))
    n_waves = max((block.wave for block in blocks), default=-1) + 1
    return CellStats(
        cell_id="c", kernel_name=kernel_name,
        grid=(len(blocks), 1, 1), block=(32, 1, 1),
        n_blocks=len(blocks), total_records=total_records, op_class_totals={},
        blocks=blocks, per_sm_resident=dict(per_sm_resident),
        per_sm_block_ids={sm: list(ids) for sm, ids in per_sm_block_ids.items()},
        n_waves=n_waves,
        atomic_records=0, atomic_top256_sector_share=0.0,
        per_wave_record_totals=per_wave_record_totals,
        per_sm_block_counts={sm: len(ids) for sm, ids in per_sm_block_ids.items()},
        stats_pass_version=2,
        serialized_bytes_exact=serialized_bytes_exact,
    )


def _stats_with_waves(n_blocks, blocks_per_wave, **kwargs):
    return _stats_for_placement(
        {0: list(range(n_blocks))}, {0: blocks_per_wave}, **kwargs)


def _split_placement_stats(n_blocks=120, **kwargs):
    midpoint = n_blocks // 2
    return _stats_for_placement(
        {0: list(range(midpoint)), 1: list(range(midpoint, n_blocks))},
        {0: 1, 1: 1}, **kwargs)


def _install_mock_pipeline(monkeypatch, orch, pass1_stats, pass2_stats):
    """Install a coherent pass1/pass2 capture pipeline and expose its calls."""
    pass2_stats = list(pass2_stats)
    selected_by_dir = {}
    selected_runs = []
    compression_calls = []

    def stats_for(path):
        path = Path(path)
        if path.name == "pass1":
            return pass1_stats
        attempt = int(path.name.rsplit("_", 1)[1])
        return pass2_stats[min(attempt - 1, len(pass2_stats) - 1)]

    def fake_run_pass(_binary, _args, out_dir, mode, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        if mode == 2:
            selected = list(kwargs["selected_blocks"])
            selected_by_dir[out_dir] = selected
            selected_runs.append(selected)

    def fake_capture_selection(path, selected):
        assert selected_by_dir[path] == list(selected)
        return hashlib.sha256(path.name.encode()).hexdigest()

    def fake_payload(path, _grid, selected, expected_serialized_bytes):
        stats = stats_for(path)
        records_by_id = {block.block_id: block.records for block in stats.blocks}
        assert selected_by_dir[path] == list(selected)
        sizes_by_id = {
            block.block_id: block.serialized_bytes for block in stats.blocks}
        assert expected_serialized_bytes == {
            block_id: sizes_by_id[block_id] for block_id in selected}
        return (
            {block_id: records_by_id[block_id] for block_id in selected},
            {block_id: sizes_by_id[block_id] for block_id in selected},
        )

    def fake_compress(path, _out):
        compression_calls.append(path)
        stats = stats_for(path)
        selected = set(selected_by_dir[path])
        raw = sum(block.serialized_bytes for block in stats.blocks
                  if block.block_id in selected)
        return SimpleNamespace(
            raw_bytes=raw, compressed_bytes=max(1, raw // 2),
            decompressed_sha256="a" * 64)

    monkeypatch.setattr(orch, "run_pass", fake_run_pass)
    monkeypatch.setattr(orch, "cell_stats_from_capture",
                        lambda path, *_a, **_k: stats_for(path))
    monkeypatch.setattr(orch, "verify_capture_selection", fake_capture_selection)
    monkeypatch.setattr(orch, "verify_pass2_payload", fake_payload)
    monkeypatch.setattr(orch, "compress_trace_dir", fake_compress)
    return SimpleNamespace(
        selected_runs=selected_runs, selected_by_dir=selected_by_dir,
        compression_calls=compression_calls, stats_for=stats_for)


def test_select_segments_picks_certainty_and_seeded_sample():
    # 120 blocks, 2 per wave -> 60 waves -> 20 segments (SEGMENT_LEN=3).
    segs = select_segments(_stats_with_waves(120, 2))
    ids = [s["id"] for s in segs]
    assert len(segs) >= MIN_SEGMENTS
    assert ids[:2] == [0, 19]            # cold-start + final certainty first
    # segment membership is explicit + non-empty
    assert all(s["block_ids"] for s in segs)


def test_select_segments_small_grid_refuses_to_floor_clamp():
    with pytest.raises(
            SelectionPlanError,
            match="below minimum|exceeds population|empty_ordinary_population"):
        select_segments(_stats_with_waves(6, 1))  # 2 segments cannot meet floor


# --- pass-2 payload verification ---

def _write_pass2_blocks(root, grid_x, block_ids, records_each=3):
    import threadblock_pb2  # type: ignore
    kd = root / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
    kd.mkdir(parents=True)
    sizes = {}
    for bid in block_ids:
        tb = threadblock_pb2.threadblock()
        tb.block_id.x, tb.block_id.y, tb.block_id.z = bid % grid_x, bid // grid_x, 0
        for _ in range(records_each):
            tb.warps[0].instructions.add()
        raw = tb.SerializeToString()
        (kd / f"b_{bid}.pb").write_bytes(raw)
        sizes[bid] = len(raw)
    return sizes


def test_verify_pass2_payload_matches_selection(tmp_path):
    from orchestrate import verify_pass2_payload
    sizes = _write_pass2_blocks(tmp_path, 8, [0, 1], records_each=3)
    actual, raw_sizes = verify_pass2_payload(
        tmp_path, (8, 1, 1), [0, 1], sizes)
    assert sorted(actual) == [0, 1]
    assert sum(actual.values()) == 6  # 2 blocks * 3 records
    assert raw_sizes == sizes


def test_verify_pass2_payload_missing_block_raises(tmp_path):
    from orchestrate import verify_pass2_payload
    sizes = _write_pass2_blocks(tmp_path, 8, [0, 1])
    sizes[2] = 1
    with pytest.raises(SamplingRefusal) as exc_info:
        verify_pass2_payload(
            tmp_path, (8, 1, 1), [0, 1, 2], sizes)  # block 2 not dumped
    assert exc_info.value.reason == "pass2_payload_membership_mismatch"


def test_verify_pass2_payload_out_of_grid_cta_raises(tmp_path):
    from orchestrate import verify_pass2_payload
    # bid 4 with grid_x=4 -> cta (0,1,0); cta.y=1 is outside a (4,1,1) grid.
    sizes = _write_pass2_blocks(tmp_path, 4, [4])
    with pytest.raises(SamplingRefusal) as exc_info:
        verify_pass2_payload(tmp_path, (4, 1, 1), [4], sizes)
    assert exc_info.value.reason == "pass2_block_out_of_grid"


def test_verify_pass2_payload_rejects_per_block_size_mismatch(tmp_path):
    from orchestrate import verify_pass2_payload
    sizes = _write_pass2_blocks(tmp_path, 8, [0, 1])
    sizes[0] += 1
    with pytest.raises(SamplingRefusal) as exc_info:
        verify_pass2_payload(tmp_path, (8, 1, 1), [0, 1], sizes)
    assert exc_info.value.reason == "pass2_serialized_bytes_mismatch"


def test_run_pass_fails_closed_on_nonzero_exit(tmp_path):
    from orchestrate import run_pass
    gcc = shutil.which("cc") or shutil.which("gcc")
    if gcc is None:
        pytest.skip("no C compiler")
    src = tmp_path / "fail.c"
    src.write_text("int main(void){return 3;}\n")
    binary = tmp_path / "fail"
    subprocess.run([gcc, str(src), "-o", str(binary)], check=True)
    with pytest.raises(RuntimeError, match="exited 3"):
        run_pass(binary, [], tmp_path / "out", mode=1)


def test_run_pass_refuses_a_preexisting_output_directory(tmp_path):
    from orchestrate import run_pass
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(SamplingRefusal) as exc_info:
        run_pass(tmp_path / "unused", [], output, mode=1)
    assert exc_info.value.reason == "capture_output_exists"


def test_orchestrate_sidecar_matches_the_seeded_population_plan(tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _stats_with_waves(120, 2)  # 20 segments: 2 certainty + 18 ordinary.
    mocked = _install_mock_pipeline(monkeypatch, orch, stats, [stats])

    sidecar = orch.orchestrate_cell(
        tmp_path / "bench", [], tmp_path / "work", seed=23,
        max_trace_bytes=1000)
    validate_sidecar(sidecar)

    assert sidecar["rng"] == {"algo": "pcg64", "seed": 23}
    assert len(sidecar["segments"]) == MIN_SEGMENTS
    assert [segment["id"] for segment in sidecar["segments"][:2]] == [0, 19]
    assert sidecar["certainty_segments"] == [0, 19]
    assert [(stratum["N_h"], stratum["n_h"], stratum["pi"])
            for stratum in sidecar["strata"]] == [
                (6, 2, pytest.approx(1 / 3)),
                (6, 2, pytest.approx(1 / 3)),
                (6, 2, pytest.approx(1 / 3)),
            ]
    assert sorted(block_id for stratum in sidecar["strata"]
                  for block_id in stratum["segment_ids"]) == sorted(
                      segment["id"] for segment in sidecar["segments"][2:])
    selected_from_sidecar = sorted(
        block_id for segment in sidecar["segments"]
        for block_id in segment["block_ids"])
    assert mocked.selected_runs == [selected_from_sidecar]
    assert sidecar["bytes"]["raw"] == len(selected_from_sidecar) * 10
    assert sidecar["tracer"]["stats_pass_version"] == 2
    assert sidecar["schema_version"] == 2
    assert sidecar["wave_model"]["per_sm_resident_blocks"] == [
        {"sm_id": 0, "resident_blocks": 2}]

    claimed_hash = sidecar["hashes"]["sidecar_sha256"]
    hash_input = copy.deepcopy(sidecar)
    hash_input["hashes"].pop("sidecar_sha256")
    assert claimed_hash == sidecar_sha256(hash_input)


def test_reconciliation_accepts_one_placement_drift_then_fixed_point(
        tmp_path, monkeypatch):
    import orchestrate as orch

    pass1 = _stats_with_waves(
        120, 2, records=100, sectors=3, serialized_bytes=10)
    accepted_pass2 = _split_placement_stats(
        records=200, sectors=20, serialized_bytes=11)
    mocked = _install_mock_pipeline(
        monkeypatch, orch, pass1, [accepted_pass2, accepted_pass2])

    sidecar = orch.orchestrate_cell(
        tmp_path / "bench", [], tmp_path / "work", seed=23,
        max_trace_bytes=1000, max_reconcile_attempts=3)
    validate_sidecar(sidecar)

    assert len(mocked.selected_runs) == 2
    assert mocked.selected_runs[0] != mocked.selected_runs[1]
    assert sidecar["selection_reconciliation"] == {
        "status": "fixed_point",
        "attempts": 2,
        "max_attempts": 3,
        "capture_stats_sha256": hashlib.sha256(
            b"pass2_attempt_2").hexdigest(),
    }

    # Every final evidence surface must come from the accepted pass-2 capture,
    # not the bootstrap placement/totals.
    selected = set(mocked.selected_runs[-1])
    assert sidecar["records"] == {
        "expected_full": accepted_pass2.total_records,
        "captured": len(selected) * 200,
    }
    assert sidecar["per_sm_slot_map"] == [
        {"sm_id": 0, "block_ids": list(range(60))},
        {"sm_id": 1, "block_ids": list(range(60, 120))},
    ]
    assert sidecar["wave_model"]["per_sm_resident_blocks"] == [
        {"sm_id": 0, "resident_blocks": 1},
        {"sm_id": 1, "resident_blocks": 1},
    ]
    assert sidecar["selection_auxiliary"] == {
        "name": "l2_sector_transactions_32b",
        "sector_bytes": 32,
        "scope": "pass2_all_blocks",
        "expected_full": 120 * 20,
        "selected": len(selected) * 20,
    }
    assert sidecar["bytes"]["raw"] == len(selected) * 11
    assert sidecar["tracer"]["stats_pass_version"] == 2


def test_reconciliation_exhaustion_has_typed_reason(tmp_path, monkeypatch):
    import orchestrate as orch

    placement_a = _stats_with_waves(120, 2)
    placement_b = _split_placement_stats()
    mocked = _install_mock_pipeline(
        monkeypatch, orch, placement_a,
        [placement_b, placement_a, placement_b])

    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            tmp_path / "bench", [], tmp_path / "work", seed=23,
            max_trace_bytes=1000, max_reconcile_attempts=3)
    assert exc_info.value.reason == "placement_reconciliation_exhausted"
    assert mocked.selected_runs[0] == mocked.selected_runs[2]
    assert mocked.selected_runs[0] != mocked.selected_runs[1]
    assert not mocked.compression_calls


def test_pass2_producer_selected_ids_mismatch_is_typed(tmp_path):
    from orchestrate import verify_capture_selection

    path = tmp_path / "extra_info" / "trace_sampling_capture.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "mode": 2,
        "selected_block_ids": [0, 2],
    }))
    with pytest.raises(SamplingRefusal) as exc_info:
        verify_capture_selection(tmp_path, [0, 1])
    assert exc_info.value.reason == "capture_selected_ids_mismatch"


def test_pass2_launch_identity_drift_is_typed(tmp_path, monkeypatch):
    import orchestrate as orch

    pass1 = _stats_with_waves(120, 2, kernel_name="kernel_a")
    pass2 = _stats_with_waves(120, 2, kernel_name="kernel_b")
    mocked = _install_mock_pipeline(monkeypatch, orch, pass1, [pass2])

    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            tmp_path / "bench", [], tmp_path / "work",
            max_trace_bytes=1000)
    assert exc_info.value.reason == "pass2_launch_identity_drift"
    assert len(mocked.selected_runs) == 1
    assert not mocked.compression_calls


def _skew_stats(*, one_extra_certainty_byte=False):
    return _stats_with_waves(
        120, 2,
        records=lambda block_id, _wave, _sm, _seq:
            10_000 if block_id == 0 else 100,
        sectors=10,
        serialized_bytes=lambda block_id, _wave, _sm, _seq: (
            11 if one_extra_certainty_byte and block_id == 0
            else 10 if block_id < 6 or block_id >= 114
            else 1),
    )


def test_exact_forty_percent_certainty_budget_is_admissible(
        tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _skew_stats()
    mocked = _install_mock_pipeline(monkeypatch, orch, stats, [stats])
    sidecar = orch.orchestrate_cell(
        tmp_path / "bench", [], tmp_path / "work",
        max_trace_bytes=300)

    assert sidecar["method"] == "wave_segment_stratified_subset"
    assert sidecar["selection_budget"]["certainty_bytes"] == 120
    assert sidecar["gates"]["certainty_budget_exceeded"] is False
    assert len(mocked.compression_calls) == 1


def test_one_byte_over_forty_percent_returns_unavailable(
        tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _skew_stats(one_extra_certainty_byte=True)
    mocked = _install_mock_pipeline(monkeypatch, orch, stats, [stats])
    sidecar = orch.orchestrate_cell(
        tmp_path / "bench", [], tmp_path / "work",
        max_trace_bytes=300)

    assert sidecar["method"] == "unavailable"
    assert sidecar["reason"] == "gate_forced"
    assert sidecar["gates"]["certainty_bytes"] == 121
    assert sidecar["gates"]["certainty_budget_exceeded"] is True
    assert "estimator" not in sidecar
    assert not mocked.compression_calls


def test_total_selected_raw_bytes_over_budget_is_a_typed_refusal(
        tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _stats_with_waves(120, 2, serialized_bytes=10)
    mocked = _install_mock_pipeline(monkeypatch, orch, stats, [stats])
    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            tmp_path / "bench", [], tmp_path / "work",
            max_trace_bytes=300)
    assert exc_info.value.reason == "selected_payload_over_budget"
    assert not mocked.compression_calls


def test_orchestrate_refuses_reusing_a_completed_work_directory(
        tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _stats_with_waves(120, 2)
    _install_mock_pipeline(monkeypatch, orch, stats, [stats])
    work = tmp_path / "work"
    orch.orchestrate_cell(
        tmp_path / "bench", [], work, max_trace_bytes=1000)
    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            tmp_path / "bench", [], work, max_trace_bytes=1000)
    assert exc_info.value.reason == "sidecar_output_exists"


def test_nonuniform_per_sm_residency_is_serialized_exactly(
        tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _stats_for_placement(
        {0: list(range(60)), 1: list(range(60, 120))},
        {0: 1, 1: 2})
    _install_mock_pipeline(monkeypatch, orch, stats, [stats])
    sidecar = orch.orchestrate_cell(
        tmp_path / "bench", [], tmp_path / "work",
        max_trace_bytes=1000)
    assert sidecar["wave_model"]["per_sm_resident_blocks"] == [
        {"sm_id": 0, "resident_blocks": 1},
        {"sm_id": 1, "resident_blocks": 2},
    ]


# --- live launch-scope regression via cell_stats_from_capture ---

@pytest.mark.skipif(not _have_live(), reason="tracer .so / GPU / nvcc absent")
def test_counting_two_launch_scoped_via_cell_stats(tmp_path):
    from counting_pass import cell_stats_from_capture
    from orchestrate import run_pass
    src = tmp_path / "twolaunch.cu"
    src.write_text(
        "#include <cuda_runtime.h>\n"
        "__global__ void k(int* g){ g[blockIdx.x*blockDim.x+threadIdx.x]=threadIdx.x; }\n"
        "int main(){ int* g; cudaMalloc(&g,2*32*sizeof(int));\n"
        "  k<<<2,32>>>(g); cudaDeviceSynchronize();\n"
        "  k<<<2,32>>>(g); cudaDeviceSynchronize(); cudaFree(g); return 0; }\n")
    binary = tmp_path / "twolaunch"
    r = subprocess.run(
        ["/usr/local/cuda-13.2/bin/nvcc", "-ccbin=g++-14", "-O3", "-std=c++17",
         "-gencode", "arch=compute_86,code=sm_86", "-I/usr/local/cuda-13.2/include",
         str(src), "-o", str(binary)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    p1 = tmp_path / "pass1"
    run_pass(binary, [], p1, mode=1)
    stats = cell_stats_from_capture(p1, "twolaunch")
    # Two launches of a 2-block grid -> the FIRST launch only (2 blocks), not 4.
    assert stats.n_blocks == 2
    assert sorted(b.block_id for b in stats.blocks) == [0, 1]


@pytest.mark.skipif(not _have_live(), reason="tracer .so / GPU / nvcc absent")
def test_live_v2_sector_and_mode2_serialized_bytes_match_payload(tmp_path):
    import threadblock_pb2  # type: ignore
    from counting_pass import analyze_trace_dir, cell_stats_from_capture
    from orchestrate import run_pass
    from replay_count_producer import first_launch_dir

    src = tmp_path / "sector_bytes.cu"
    src.write_text(
        "#include <cuda_runtime.h>\n"
        "__global__ void k(int* g){\n"
        "  unsigned i=blockIdx.x*blockDim.x+threadIdx.x; g[i]=(int)i;\n"
        "}\n"
        "int main(){ int* g; cudaMalloc(&g,2*32*sizeof(int));\n"
        "  k<<<2,32>>>(g); cudaDeviceSynchronize(); cudaFree(g); return 0; }\n")
    binary = tmp_path / "sector_bytes"
    result = subprocess.run(
        ["/usr/local/cuda-13.2/bin/nvcc", "-ccbin=g++-14", "-O3", "-std=c++17",
         "-gencode", "arch=compute_86,code=sm_86", "-I/usr/local/cuda-13.2/include",
         str(src), "-o", str(binary)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()

    pass1 = tmp_path / "pass1_v2"
    run_pass(binary, [], pass1, mode=1)
    pass1_capture = json.loads(
        (pass1 / "extra_info" / "trace_sampling_capture.json").read_text())
    pass1_stats = cell_stats_from_capture(pass1, "sector_bytes")
    assert pass1_capture["stats_pass_version"] == 2
    assert pass1_capture["sector_accesses_valid"] is True
    assert pass1_capture["serialized_bytes_exact"] is False
    assert pass1_stats.stats_pass_version == 2
    assert {block.block_id: block.sector_accesses
            for block in pass1_stats.blocks} == {0: 4, 1: 4}

    pass2 = tmp_path / "pass2_v2"
    run_pass(binary, [], pass2, mode=2, selected_blocks=[0])
    pass2_capture = json.loads(
        (pass2 / "extra_info" / "trace_sampling_capture.json").read_text())
    pass2_stats = cell_stats_from_capture(pass2, "sector_bytes")
    assert pass2_capture["stats_pass_version"] == 2
    assert pass2_capture["sector_accesses_valid"] is True
    assert pass2_capture["serialized_bytes_exact"] is True
    assert pass2_capture["selected_block_ids"] == [0]
    assert pass2_stats.serialized_bytes_exact
    assert {block.block_id: block.sector_accesses
            for block in pass2_stats.blocks} == {0: 4, 1: 4}

    first_launch, _ = first_launch_dir(pass2)
    payloads = list(first_launch.rglob("*.pb"))
    assert len(payloads) == 1
    tb = threadblock_pb2.threadblock()
    raw = payloads[0].read_bytes()
    tb.ParseFromString(raw)
    selected_id = int(tb.block_id.x)
    assert selected_id == 0
    by_id = {block.block_id: block for block in pass2_stats.blocks}
    assert by_id[selected_id].serialized_bytes == len(raw)
    assert by_id[1].serialized_bytes > 0  # measured before unselected clear

    payload_records = sum(
        len(tb.warps[warp_id].instructions) for warp_id in tb.warps)
    pass1_records = {block.block_id: block.records for block in pass1_stats.blocks}
    pass2_records = {block.block_id: block.records for block in pass2_stats.blocks}
    assert pass1_records == pass2_records
    assert payload_records == pass2_records[selected_id]
    assert analyze_trace_dir(pass2, "sector_bytes").total_sector_accesses == 4


@pytest.mark.skipif(not _have_live(), reason="tracer .so / GPU / nvcc absent")
def test_live_cross_allocation_record_invalidates_sector_evidence(tmp_path):
    from counting_pass import cell_stats_from_capture
    from orchestrate import run_pass

    src = tmp_path / "cross_allocation.cu"
    src.write_text(
        "#include <cuda_runtime.h>\n"
        "__global__ void k(int** choices){\n"
        "  unsigned lane=threadIdx.x; int* p=choices[lane >> 4];\n"
        "  p[lane & 15]=(int)lane;\n"
        "}\n"
        "int main(){ int *a,*b,**choices; cudaMalloc(&a,16*sizeof(int));\n"
        "  cudaMalloc(&b,16*sizeof(int)); cudaMalloc(&choices,2*sizeof(int*));\n"
        "  int* h[2]={a,b}; cudaMemcpy(choices,h,sizeof(h),cudaMemcpyHostToDevice);\n"
        "  k<<<1,32>>>(choices); cudaDeviceSynchronize();\n"
        "  cudaFree(choices); cudaFree(b); cudaFree(a); return 0; }\n")
    binary = tmp_path / "cross_allocation"
    result = subprocess.run(
        ["/usr/local/cuda-13.2/bin/nvcc", "-ccbin=g++-14", "-O3", "-std=c++17",
         "-gencode", "arch=compute_86,code=sm_86", "-I/usr/local/cuda-13.2/include",
         str(src), "-o", str(binary)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()

    capture_dir = tmp_path / "cross_pass1"
    run_pass(binary, [], capture_dir, mode=1)
    capture = json.loads(
        (capture_dir / "extra_info" / "trace_sampling_capture.json").read_text())
    assert capture["stats_pass_version"] == 2
    assert capture["sector_accesses_valid"] is False
    with pytest.raises(ValueError, match="sector_accesses invalid"):
        cell_stats_from_capture(capture_dir, "cross_allocation")


# --- live two-pass integration (guarded) ---

@pytest.mark.skipif(not _have_live(), reason="tracer .so / GPU / nvcc absent")
def test_orchestrate_histogram_routes_to_unavailable(tmp_path):
    # The atomic histogram is atomic-hotline-forbidden -> typed unavailable.
    cu = _REPO / "tools" / "local_labels" / "canary" / "ac4_global_atomic_histogram.cu"
    binary = tmp_path / "histo"
    r = subprocess.run(
        ["/usr/local/cuda-13.2/bin/nvcc", "-ccbin=g++-14", "-O3", "-std=c++17",
         "-gencode", "arch=compute_86,code=sm_86", "-I/usr/local/cuda-13.2/include",
         str(cu), "-o", str(binary)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    sidecar = orchestrate_cell(binary, [], tmp_path / "work")
    validate_sidecar(sidecar)
    assert sidecar["method"] == "unavailable"
    # 32 blocks across 84 SMs -> too few waves: the histogram fails the wave_gate
    # STRUCTURALLY before the atomic_gate (exactly as the design predicts), so
    # either forbidding verdict is correct; both route to typed unavailable.
    assert sidecar["gates"]["verdict"] in {"wave_forbid", "atomic_forbid"}
    assert "estimator" not in sidecar  # unavailable forbids estimator point outputs
