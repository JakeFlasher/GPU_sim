"""Orchestration: seeded segment selection + a live two-pass integration."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

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


def _install_mock_pipeline(
        monkeypatch, orch, pass1_stats, pass2_stats, *, retain_inputs=False,
        input_exit_hook=None):
    """Install a coherent pass1/pass2 capture pipeline and expose its calls."""
    pass2_stats = list(pass2_stats)
    selected_by_dir = {}
    selected_runs = []
    compression_calls = []
    retained_pairs = []
    retained_caps = []

    def stats_for(path):
        path = Path(path)
        if path.name == "pass1":
            return pass1_stats
        attempt = int(path.name.rsplit("_", 1)[1])
        return pass2_stats[min(attempt - 1, len(pass2_stats) - 1)]

    def fake_run_pass(_binary, _args, out_dir, mode, **kwargs):
        retained_pairs.append((
            kwargs.get("retained_binary"),
            kwargs.get("retained_tracer"),
        ))
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
            decompressed_sha256="a" * 64,
            per_block_raw_sha256_manifest=(
                "trace-payload-file-manifest-v1\n"
                f"threadblocks/device_0/stream_0/kernel_1/mock.pb\t{raw}\t"
                + "a" * 64 + "\n"),
        )

    if not retain_inputs:
        class FakeRetained:
            def __init__(self, path, descriptor):
                self.path = Path(path)
                self._descriptor = descriptor

            @property
            def descriptor(self):
                if self._descriptor < 0:
                    raise immutable.ImmutableArtifactError("retained input closed")
                return self._descriptor

            @property
            def proc_path(self):
                return f"/proc/self/fd/{self.descriptor}"

            def verify_unchanged(self):
                self.descriptor

            def __enter__(self):
                return self

            def __exit__(self, _type, _value, _traceback):
                self._descriptor = -1
                if input_exit_hook is not None:
                    input_exit_hook(self.path)
                return False

        def fake_binary_opener(path):
            retained = FakeRetained(path, 901)
            retained_caps.append(retained)
            return retained

        def fake_tracer_opener(path):
            if Path(path).name == "trace_sampling.json":
                return immutable.open_retained_regular(Path(path))
            retained = FakeRetained(path, 902)
            retained_caps.append(retained)
            return retained

        monkeypatch.setattr(orch, "open_retained_executable", fake_binary_opener)
        monkeypatch.setattr(orch, "open_retained_regular", fake_tracer_opener)

    monkeypatch.setattr(orch, "run_pass", fake_run_pass)
    monkeypatch.setattr(orch, "cell_stats_from_capture",
                        lambda path, *_a, **_k: stats_for(path))
    monkeypatch.setattr(orch, "verify_capture_selection", fake_capture_selection)
    monkeypatch.setattr(orch, "verify_pass2_payload", fake_payload)
    monkeypatch.setattr(orch, "compress_trace_dir", fake_compress)
    return SimpleNamespace(
        selected_runs=selected_runs, selected_by_dir=selected_by_dir,
        compression_calls=compression_calls, stats_for=stats_for,
        retained_pairs=retained_pairs, retained_caps=retained_caps)


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
    threadblock_pb2 = load_dynamic_trace_bindings().threadblock_pb2
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


def _write_retained_sampling_inputs(tmp_path):
    binary = tmp_path / "bench"
    binary.write_bytes(b"exact executable generation")
    binary.chmod(0o755)
    tracer = tmp_path / "tracer_tool.so"
    tracer.write_bytes(b"exact tracer generation")
    tracer.chmod(0o644)
    return binary, tracer


def test_run_pass_executes_exact_retained_descriptors(tmp_path, monkeypatch):
    import orchestrate as orch

    binary, tracer = _write_retained_sampling_inputs(tmp_path)
    output = tmp_path / "capture"
    observed = {}

    def fake_runtime(path, *, pb_lib, retained_descriptor):
        observed["runtime"] = (path, pb_lib, retained_descriptor)
        return {}

    def fake_subprocess_run(command, **kwargs):
        observed["command"] = list(command)
        observed["kwargs"] = kwargs
        (output / "dynamic_trace.pb").write_bytes(b"trace")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(orch, "tracer_runtime_env", fake_runtime)
    monkeypatch.setattr(orch.subprocess, "run", fake_subprocess_run)

    orch.run_pass(
        binary,
        ["arg"],
        output,
        mode=1,
        tracer_so=tracer,
        pb_lib=tmp_path / "pb-lib",
    )

    binary_proc, app_arg = observed["command"]
    tracer_proc = observed["kwargs"]["env"]["CUDA_INJECTION64_PATH"]
    binary_fd = int(binary_proc.rsplit("/", 1)[1])
    tracer_fd = int(tracer_proc.rsplit("/", 1)[1])
    assert binary_proc == f"/proc/self/fd/{binary_fd}"
    assert tracer_proc == f"/proc/self/fd/{tracer_fd}"
    assert app_arg == "arg"
    assert observed["kwargs"]["pass_fds"] == (binary_fd, tracer_fd)
    assert observed["runtime"][2] == tracer_fd
    assert observed["runtime"][0] == tracer.absolute()
    assert str(binary) not in observed["command"]
    assert observed["kwargs"]["cwd"] == str(binary.parent)


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
    assert len(mocked.retained_pairs) == 2
    assert mocked.retained_pairs[0][0] is mocked.retained_pairs[1][0]
    assert mocked.retained_pairs[0][1] is mocked.retained_pairs[1][1]
    assert all(cap._descriptor == -1 for cap in mocked.retained_caps)
    assert sidecar["bytes"]["raw"] == len(selected_from_sidecar) * 10
    assert sidecar["tracer"]["stats_pass_version"] == 2
    assert sidecar["schema_version"] == 2
    assert sidecar["wave_model"]["per_sm_resident_blocks"] == [
        {"sm_id": 0, "resident_blocks": 2}]

    claimed_hash = sidecar["hashes"]["sidecar_sha256"]
    hash_input = copy.deepcopy(sidecar)
    hash_input["hashes"].pop("sidecar_sha256")
    assert claimed_hash == sidecar_sha256(hash_input)


def _record_real_retained_inputs(monkeypatch, orch):
    opened = []
    real_binary_open = orch.open_retained_executable
    real_tracer_open = orch.open_retained_regular

    def open_binary(path):
        retained = real_binary_open(path)
        opened.append(retained)
        return retained

    def open_tracer(path):
        retained = real_tracer_open(path)
        if Path(path).name != "trace_sampling.json":
            opened.append(retained)
        return retained

    monkeypatch.setattr(orch, "open_retained_executable", open_binary)
    monkeypatch.setattr(orch, "open_retained_regular", open_tracer)
    return opened


def _assert_retained_inputs_closed(opened):
    assert len(opened) == 2
    for retained in opened:
        with pytest.raises(immutable.ImmutableArtifactError, match="closed"):
            retained.descriptor


def test_orchestrate_closes_one_real_descriptor_pair_after_both_passes(
        tmp_path, monkeypatch):
    import orchestrate as orch

    binary, tracer = _write_retained_sampling_inputs(tmp_path)
    stats = _stats_with_waves(120, 2)
    mocked = _install_mock_pipeline(
        monkeypatch, orch, stats, [stats], retain_inputs=True)
    opened = _record_real_retained_inputs(monkeypatch, orch)

    sidecar = orch.orchestrate_cell(
        binary, [], tmp_path / "work", tracer_so=tracer,
        max_trace_bytes=1000)

    assert sidecar["method"] == "wave_segment_stratified_subset"
    assert len(mocked.retained_pairs) == 2
    assert mocked.retained_pairs[0] == mocked.retained_pairs[1]
    assert mocked.retained_pairs[0] == tuple(opened)
    _assert_retained_inputs_closed(opened)


@pytest.mark.parametrize("mutation", ("same_byte_replacement", "aba"))
def test_orchestrate_rejects_binary_generation_change_after_pass1(
        tmp_path, monkeypatch, mutation):
    import orchestrate as orch

    binary, tracer = _write_retained_sampling_inputs(tmp_path)
    stats = _stats_with_waves(120, 2)
    _install_mock_pipeline(
        monkeypatch, orch, stats, [stats], retain_inputs=True)
    opened = _record_real_retained_inputs(monkeypatch, orch)
    mocked_run_pass = orch.run_pass
    mutated = False

    def mutate_after_pass1(*args, **kwargs):
        nonlocal mutated
        mocked_run_pass(*args, **kwargs)
        mode = kwargs.get("mode", args[3] if len(args) > 3 else None)
        if mode != 1 or mutated:
            return
        mutated = True
        if mutation == "same_byte_replacement":
            replacement = binary.with_name(".binary-replacement")
            replacement.write_bytes(b"exact executable generation")
            replacement.chmod(0o755)
            os.replace(replacement, binary)
        else:
            parked = binary.with_name(".binary-parked")
            os.replace(binary, parked)
            os.replace(parked, binary)

    monkeypatch.setattr(orch, "run_pass", mutate_after_pass1)

    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            binary, [], tmp_path / "work", tracer_so=tracer,
            max_trace_bytes=1000)

    assert mutated is True
    assert exc_info.value.reason == "sampling_input_invalid"
    assert not (tmp_path / "work" / "trace_sampling.json").exists()
    _assert_retained_inputs_closed(opened)


def test_orchestrate_rejects_tracer_replacement_before_final_publication(
        tmp_path, monkeypatch):
    import orchestrate as orch

    binary, tracer = _write_retained_sampling_inputs(tmp_path)
    stats = _stats_with_waves(120, 2)
    _install_mock_pipeline(
        monkeypatch, orch, stats, [stats], retain_inputs=True)
    opened = _record_real_retained_inputs(monkeypatch, orch)
    original_build = orch.build_sidecar
    mutated = False

    def build_then_replace_tracer(*args, **kwargs):
        nonlocal mutated
        sidecar = original_build(*args, **kwargs)
        if not mutated:
            mutated = True
            replacement = tracer.with_name(".tracer-replacement")
            replacement.write_bytes(b"exact tracer generation")
            replacement.chmod(0o644)
            os.replace(replacement, tracer)
        return sidecar

    monkeypatch.setattr(orch, "build_sidecar", build_then_replace_tracer)

    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            binary, [], tmp_path / "work", tracer_so=tracer,
            max_trace_bytes=1000)

    assert mutated is True
    assert exc_info.value.reason == "sampling_input_invalid"
    assert not (tmp_path / "work" / "trace_sampling.json").exists()
    _assert_retained_inputs_closed(opened)


def test_orchestrate_rejects_binary_replacement_during_sidecar_publication(
        tmp_path, monkeypatch):
    import orchestrate as orch

    binary, tracer = _write_retained_sampling_inputs(tmp_path)
    stats = _stats_with_waves(120, 2)
    _install_mock_pipeline(
        monkeypatch, orch, stats, [stats], retain_inputs=True)
    opened = _record_real_retained_inputs(monkeypatch, orch)
    original_write = orch.write_sidecar
    mutated = False

    def write_then_replace_binary(
            sidecar, path, *, publication_state=None):
        nonlocal mutated
        original_write(
            sidecar, path, publication_state=publication_state)
        replacement = binary.with_name(".publication-binary-replacement")
        replacement.write_bytes(b"exact executable generation")
        replacement.chmod(0o755)
        os.replace(replacement, binary)
        mutated = True

    monkeypatch.setattr(orch, "write_sidecar", write_then_replace_binary)

    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            binary, [], tmp_path / "work", tracer_so=tracer,
            max_trace_bytes=1000)

    out = tmp_path / "work" / "trace_sampling.json"
    assert mutated is True
    assert exc_info.value.reason == "sampling_input_invalid"
    assert not out.exists()
    retired = tuple(out.parent.glob(".trace_sampling.json.*.retired"))
    assert len(retired) == 1
    validate_sidecar(json.loads(retired[0].read_text(encoding="utf-8")))
    _assert_retained_inputs_closed(opened)


def test_writer_publish_then_raise_retires_exact_sidecar(
        tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _stats_with_waves(120, 2)
    mocked = _install_mock_pipeline(monkeypatch, orch, stats, [stats])
    original_write = orch.write_sidecar

    def publish_then_raise(
            sidecar, path, *, publication_state=None):
        original_write(
            sidecar, path, publication_state=publication_state)
        raise RuntimeError("writer failed after publication")

    monkeypatch.setattr(orch, "write_sidecar", publish_then_raise)

    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            tmp_path / "bench", [], tmp_path / "work",
            max_trace_bytes=1000)

    out = tmp_path / "work" / "trace_sampling.json"
    assert exc_info.value.reason == "sidecar_publication_failed"
    assert exc_info.value.evidence_state == "expected_payload_retired"
    assert "sidecar_evidence_state=expected_payload_retired" in str(
        exc_info.value)
    assert not out.exists()
    retired = tuple(out.parent.glob(".trace_sampling.json.*.retired"))
    assert len(retired) == 1
    validate_sidecar(json.loads(retired[0].read_text(encoding="utf-8")))
    assert all(cap._descriptor == -1 for cap in mocked.retained_caps)


def test_writer_publish_replace_then_raise_preserves_replacement(
        tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _stats_with_waves(120, 2)
    mocked = _install_mock_pipeline(monkeypatch, orch, stats, [stats])
    original_write = orch.write_sidecar
    replacement_payload = b'{"concurrent":"replacement"}\n'

    def publish_replace_then_raise(
            sidecar, path, *, publication_state=None):
        original_write(
            sidecar, path, publication_state=publication_state)
        replacement = path.with_name(".concurrent-sidecar")
        replacement.write_bytes(replacement_payload)
        os.replace(replacement, path)
        raise RuntimeError("writer failed after replacement")

    monkeypatch.setattr(
        orch, "write_sidecar", publish_replace_then_raise)

    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            tmp_path / "bench", [], tmp_path / "work",
            max_trace_bytes=1000)

    out = tmp_path / "work" / "trace_sampling.json"
    assert exc_info.value.reason == "sidecar_publication_failed"
    assert exc_info.value.evidence_state == (
        "retained_payload_salvaged_active_locator_untouched")
    assert out.read_bytes() == replacement_payload
    assert not tuple(out.parent.glob(".trace_sampling.json.*.retired"))
    evidence = tuple(out.parent.glob(".failed-regular-*"))
    assert len(evidence) == 1
    validate_sidecar(json.loads(evidence[0].read_text(encoding="utf-8")))
    assert all(cap._descriptor == -1 for cap in mocked.retained_caps)


def test_writer_same_byte_replace_then_raise_preserves_b_and_a_evidence(
        tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _stats_with_waves(120, 2)
    mocked = _install_mock_pipeline(monkeypatch, orch, stats, [stats])
    original_write = orch.write_sidecar
    original_retire = orch.retire_regular_file
    writer_snapshot = None
    replacement_identity = None
    retirement_authority = {}

    def publish_same_byte_replace_then_raise(
            sidecar, path, *, publication_state=None):
        nonlocal writer_snapshot, replacement_identity
        assert publication_state is not None
        original_write(
            sidecar, path, publication_state=publication_state)
        writer_snapshot = publication_state.snapshot
        assert writer_snapshot is not None
        payload = path.read_bytes()
        replacement = path.with_name(".same-byte-writer-replacement")
        replacement.write_bytes(payload)
        os.replace(replacement, path)
        metadata = path.stat()
        replacement_identity = (metadata.st_dev, metadata.st_ino)
        assert replacement_identity != writer_snapshot[:2]
        raise RuntimeError("writer failed after same-byte replacement")

    def record_retirement(
            path, *, expected_payload=None, expected_snapshot=None):
        retirement_authority["payload"] = expected_payload
        retirement_authority["snapshot"] = expected_snapshot
        return original_retire(
            path,
            expected_payload=expected_payload,
            expected_snapshot=expected_snapshot,
        )

    monkeypatch.setattr(
        orch, "write_sidecar", publish_same_byte_replace_then_raise)
    monkeypatch.setattr(orch, "retire_regular_file", record_retirement)

    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            tmp_path / "bench", [], tmp_path / "work",
            max_trace_bytes=1000)

    out = tmp_path / "work" / "trace_sampling.json"
    assert writer_snapshot is not None
    assert replacement_identity is not None
    assert exc_info.value.reason == "sidecar_publication_failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "same-byte replacement" in str(exc_info.value.__cause__)
    assert exc_info.value.evidence_state == (
        "retained_payload_salvaged_active_locator_untouched")
    assert retirement_authority["snapshot"] == writer_snapshot
    assert retirement_authority["payload"] == out.read_bytes()
    assert (out.stat().st_dev, out.stat().st_ino) == replacement_identity
    assert not tuple(out.parent.glob(".trace_sampling.json.*.retired"))
    evidence = tuple(out.parent.glob(".failed-regular-*"))
    assert len(evidence) == 1
    assert evidence[0].read_bytes() == out.read_bytes()
    validate_sidecar(json.loads(evidence[0].read_text(encoding="utf-8")))
    assert all(cap._descriptor == -1 for cap in mocked.retained_caps)


def test_same_byte_sidecar_replacement_during_input_close_is_preserved(
        tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _stats_with_waves(120, 2)
    out = tmp_path / "work" / "trace_sampling.json"
    replacement_identity = None

    def replace_once(_input_path):
        nonlocal replacement_identity
        if replacement_identity is not None:
            return
        payload = out.read_bytes()
        replacement = out.with_name(".same-byte-sidecar")
        replacement.write_bytes(payload)
        os.replace(replacement, out)
        metadata = out.stat()
        replacement_identity = (metadata.st_dev, metadata.st_ino)

    mocked = _install_mock_pipeline(
        monkeypatch, orch, stats, [stats], input_exit_hook=replace_once)

    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            tmp_path / "bench", [], tmp_path / "work",
            max_trace_bytes=1000)

    assert replacement_identity is not None
    assert exc_info.value.reason == "sidecar_publication_changed"
    assert exc_info.value.evidence_state == (
        "retained_payload_salvaged_active_locator_untouched")
    assert (out.stat().st_dev, out.stat().st_ino) == replacement_identity
    validate_sidecar(json.loads(out.read_text(encoding="utf-8")))
    assert not tuple(out.parent.glob(".trace_sampling.json.*.retired"))
    evidence = tuple(out.parent.glob(".failed-regular-*"))
    assert len(evidence) == 1
    assert evidence[0].read_bytes() == out.read_bytes()
    assert all(cap._descriptor == -1 for cap in mocked.retained_caps)


def test_sidecar_replacement_during_return_boundary_is_preserved(
        tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _stats_with_waves(120, 2)
    mocked = _install_mock_pipeline(monkeypatch, orch, stats, [stats])
    out = tmp_path / "work" / "trace_sampling.json"
    original_opener = orch.open_retained_regular
    replacement_payload = b'{"return_boundary":"replacement"}\n'
    replaced = False

    class ReplaceBeforeExit:
        def __init__(self, retained):
            self._retained = retained

        def __getattr__(self, name):
            return getattr(self._retained, name)

        def __enter__(self):
            self._retained.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            nonlocal replaced
            if not replaced:
                replacement = out.with_name(".return-boundary-sidecar")
                replacement.write_bytes(replacement_payload)
                os.replace(replacement, out)
                replaced = True
            return self._retained.__exit__(exc_type, exc, traceback)

    def retain_with_exit_replacement(path):
        retained = original_opener(path)
        if Path(path) == out:
            return ReplaceBeforeExit(retained)
        return retained

    monkeypatch.setattr(
        orch, "open_retained_regular", retain_with_exit_replacement)

    with pytest.raises(SamplingRefusal) as exc_info:
        orch.orchestrate_cell(
            tmp_path / "bench", [], tmp_path / "work",
            max_trace_bytes=1000)

    assert replaced is True
    assert exc_info.value.reason == "sidecar_publication_changed"
    assert exc_info.value.evidence_state == (
        "retained_payload_salvaged_active_locator_untouched")
    assert out.read_bytes() == replacement_payload
    assert not tuple(out.parent.glob(".trace_sampling.json.*.retired"))
    evidence = tuple(out.parent.glob(".failed-regular-*"))
    assert len(evidence) == 1
    validate_sidecar(json.loads(evidence[0].read_text(encoding="utf-8")))
    assert all(cap._descriptor == -1 for cap in mocked.retained_caps)


def test_success_returns_exact_active_sidecar_bytes(tmp_path, monkeypatch):
    import orchestrate as orch

    stats = _stats_with_waves(120, 2)
    _install_mock_pipeline(monkeypatch, orch, stats, [stats])

    sidecar = orch.orchestrate_cell(
        tmp_path / "bench", [], tmp_path / "work",
        max_trace_bytes=1000)

    expected = (json.dumps(sidecar, indent=2) + "\n").encode("utf-8")
    assert (tmp_path / "work" / "trace_sampling.json").read_bytes() == expected


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
    threadblock_pb2 = load_dynamic_trace_bindings().threadblock_pb2
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
