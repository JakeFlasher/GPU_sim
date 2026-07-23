"""Counting pass: pure skew/wave helpers + decode of a real captured trace."""
from pathlib import Path

import pytest

from counting_pass import (ATOMIC, ASYNC_COPY, BlockStat, _gini,
                           _reconstruct_waves, _resident_from_slots,
                           _top_fraction_share, analyze_trace_dir)

_REPO = Path(__file__).resolve().parents[3]
_FIXTURE = _REPO / "sim" / "trace" / "tests" / "fixtures" / "addr_carry_canary"


def test_gini_uniform_is_zero():
    assert _gini([100, 100, 100, 100]) == pytest.approx(0.0, abs=1e-9)


def test_gini_concentrated_is_high():
    assert _gini([1000] + [1] * 99) > 0.89


def test_top_fraction_share():
    assert _top_fraction_share([100, 100, 100, 100], 0.01) == pytest.approx(0.25)
    assert _top_fraction_share([970] + [1] * 30, 0.01) > 0.9


def test_resident_from_slots():
    assert _resident_from_slots(max_warp_slot=7, warps_per_block=1) == 8
    assert _resident_from_slots(max_warp_slot=7, warps_per_block=2) == 4
    assert _resident_from_slots(max_warp_slot=-1, warps_per_block=3) == 1


def test_wave_reconstruction_on_launch_index():
    blocks = [BlockStat(block_id=i, sm_id=0, records=10, warps_per_block=1,
                        max_warp_slot=1) for i in range(4)]
    resident, block_ids, n_waves = _reconstruct_waves(blocks)
    assert resident[0] == 2  # cap 2 / 1 warp-per-block
    assert n_waves == 2      # 4 blocks / 2 resident
    assert sorted(b.wave for b in blocks) == [0, 0, 1, 1]


def _write_two_launch_trace(root: Path) -> None:
    import threadblock_pb2  # type: ignore
    import trace_pb2  # type: ignore

    t = trace_pb2.Trace()
    st = t.gpu_device[0].streams[0]
    for kid in (1, 2):
        k = st.kernels.add()
        k.id = kid
        k.name = "kernel"
        k.grid_dim.x, k.grid_dim.y, k.grid_dim.z = 2, 1, 1
        k.block_dim.x, k.block_dim.y, k.block_dim.z = 32, 1, 1
    (root / "dynamic_trace.pb").write_bytes(t.SerializeToString())
    for kid in (1, 2):
        kd = root / "threadblocks" / "device_0" / "stream_0" / f"kernel_{kid}"
        kd.mkdir(parents=True)
        for bx in range(2):
            tb = threadblock_pb2.threadblock()
            tb.block_id.x, tb.block_id.y, tb.block_id.z = bx, 0, 0
            inst = tb.warps[0].instructions.add()
            inst.sm_id, inst.warpid_sm = 0, 0
            (kd / f"b_{bx}.pb").write_bytes(tb.SerializeToString())


def _write_sector_trace(root: Path, records) -> Path:
    """Write one block; each record is (class, width, space, operands, masks)."""
    import threadblock_pb2  # type: ignore
    import trace_pb2  # type: ignore

    t = trace_pb2.Trace()
    k = t.gpu_device[0].streams[0].kernels.add()
    k.id, k.name = 1, "sector_kernel"
    k.grid_dim.x, k.grid_dim.y, k.grid_dim.z = 1, 1, 1
    k.block_dim.x, k.block_dim.y, k.block_dim.z = 32, 1, 1
    (root / "dynamic_trace.pb").write_bytes(t.SerializeToString())
    kd = root / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
    kd.mkdir(parents=True)
    tb = threadblock_pb2.threadblock()
    tb.block_id.x, tb.block_id.y, tb.block_id.z = 0, 0, 0
    for pc, (op_class, width, space, operands, active, predicate) in enumerate(records):
        inst = tb.warps[0].instructions.add()
        inst.pc = pc
        inst.sm_id, inst.warpid_sm = 0, 0
        inst.active_mask, inst.predicate_mask = active, predicate
        inst.memory_op_class = op_class
        inst.data_width = width
        for lane_addresses in operands:
            addr = inst.addresses.add()
            addr.data_width = width
            addr.compression_format = 0  # list_all
            if space is not None:
                addr.address_space = space
            values = [-1] * 32
            for lane, value in lane_addresses.items():
                values[lane] = value
            addr.addrs.extend(values)
    path = kd / "b_0.pb"
    path.write_bytes(tb.SerializeToString())
    return path


def test_counting_is_scoped_to_first_launch(tmp_path):
    # Two launches of a 2-block grid; only the first launch must be counted.
    _write_two_launch_trace(tmp_path)
    stats = analyze_trace_dir(tmp_path, cell_id="two_launch")
    assert stats.n_blocks == 2          # NOT 4 (both launches)
    assert stats.total_records == 2


def test_full_trace_sector_accesses_are_per_operand_deduplicated(tmp_path):
    records = [
        # Equal sectors deduplicate within an operand, but the second operand is
        # a distinct transaction record and contributes independently: 1 + 1.
        (1, 4, 1, [{0: 0, 1: 4}, {0: 0, 1: 4}], 0b11, 0b11),
        # A misaligned 16-byte lane interval spans sectors 0 and 1: 2.
        (1, 16, 1, [{0: 24}], 0b1, 0b1),
        # Addressed global atomic: 1; addressed shared atomic: 0.
        (6, 4, 1, [{0: 64, 1: 68}], 0b11, 0b11),
        (6, 4, 2, [{0: 96}], 0b1, 0b1),
        # ASYNC_COPY global source crosses a sector boundary: 2.
        (8, 8, 1, [{0: 124}], 0b1, 0b1),
        # Lane 1 is active but predicated off, so only lane 0 contributes: 1.
        (1, 4, 1, [{0: 160, 1: 192}], 0b11, 0b01),
    ]
    block_path = _write_sector_trace(tmp_path, records)
    stats = analyze_trace_dir(tmp_path, cell_id="sectors")
    assert stats.blocks[0].sector_accesses == 8
    assert stats.total_sector_accesses == 8
    assert stats.blocks[0].serialized_bytes == block_path.stat().st_size
    # The pre-existing atomic sketch remains record/lane based.
    assert stats.atomic_records == 2


def test_full_trace_async_copy_missing_global_source_space_is_rejected(tmp_path):
    _write_sector_trace(
        tmp_path, [(8, 4, None, [{0: 0}], 0b1, 0b1)])
    with pytest.raises(ValueError, match="ASYNC_COPY source"):
        analyze_trace_dir(tmp_path, cell_id="bad_async")


def test_full_trace_global_missing_space_uses_class_fallback(tmp_path):
    _write_sector_trace(
        tmp_path, [(1, 4, None, [{0: 0, 1: 4}], 0b11, 0b11)])
    stats = analyze_trace_dir(tmp_path, cell_id="legacy_global")
    assert stats.total_sector_accesses == 1


def test_analyze_trace_dir_rejects_per_axis_out_of_grid_cta(tmp_path):
    import threadblock_pb2  # type: ignore
    import trace_pb2  # type: ignore

    t = trace_pb2.Trace()
    k = t.gpu_device[0].streams[0].kernels.add()
    k.id, k.name = 1, "k"
    k.grid_dim.x, k.grid_dim.y, k.grid_dim.z = 2, 2, 1
    k.block_dim.x, k.block_dim.y, k.block_dim.z = 32, 1, 1
    (tmp_path / "dynamic_trace.pb").write_bytes(t.SerializeToString())
    kd = tmp_path / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
    kd.mkdir(parents=True)
    # A valid CTA (0,0,0) + an out-of-grid CTA (2,0,0): x=2 >= gx=2, but it
    # flattens to index 2 (inside [0,4)) so a flattened-only check would alias it.
    for (bx, by) in [(0, 0), (2, 0)]:
        tb = threadblock_pb2.threadblock()
        tb.block_id.x, tb.block_id.y, tb.block_id.z = bx, by, 0
        inst = tb.warps[0].instructions.add()
        inst.sm_id, inst.warpid_sm = 0, 0
        (kd / f"b_{bx}_{by}.pb").write_bytes(tb.SerializeToString())
    with pytest.raises(ValueError, match="outside grid"):
        analyze_trace_dir(tmp_path, cell_id="c")


def _write_capture(root, *, grid, block, launch=(0, 0, 1), cap_launch=None,
                   cap_grid=None, cap_block=None, ctas=None, omit=None,
                   mode=1, stats_pass_version=2, block_overrides=None,
                   block_omit=None):
    import json as _json
    import trace_pb2  # type: ignore

    t = trace_pb2.Trace()
    st = t.gpu_device[launch[0]].streams[launch[1]]
    k = st.kernels.add()
    k.id = launch[2]  # the dynamic_trace launch always carries integer identity
    k.grid_dim.x, k.grid_dim.y, k.grid_dim.z = grid
    k.block_dim.x, k.block_dim.y, k.block_dim.z = block
    (root / "dynamic_trace.pb").write_bytes(t.SerializeToString())
    ei = root / "extra_info"
    ei.mkdir(parents=True)
    cl = cap_launch or launch
    cg, cb = cap_grid or grid, cap_block or block
    if ctas is None:
        ctas = [(x, y, z) for z in range(grid[2])
                for y in range(grid[1]) for x in range(grid[0])]
    blocks = [{"cta": list(c), "records": 10, "sm_id": 0,
               "max_warpid_sm": 0, "atomic_records": 0,
               "sector_accesses": i + 1,
               "serialized_bytes": (100 + i) if mode == 2 else 0}
              for i, c in enumerate(ctas)]
    if blocks and block_overrides:
        blocks[0].update(block_overrides)
    if blocks:
        for key in (block_omit or []):
            blocks[0].pop(key, None)
    cap = {
        "mode": mode, "tracer_version": 5,
        "stats_pass_version": stats_pass_version,
        "launch": {"device": cl[0], "stream": cl[1], "kernel": cl[2]},
        "grid_dim": {"x": cg[0], "y": cg[1], "z": cg[2]},
        "block_dim": {"x": cb[0], "y": cb[1], "z": cb[2]},
        "atomic_sector_total": 0, "atomic_top256_touches": 0,
        "atomic_sectors_overflow": False,
        "sector_accesses_valid": True,
        "serialized_bytes_exact": mode == 2,
        "blocks": blocks,
        "selected_block_ids": [],
    }
    for key in (omit or []):
        cap.pop(key, None)
    (ei / "trace_sampling_capture.json").write_text(_json.dumps(cap))


def test_cell_stats_from_capture_ok(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(4, 1, 1), block=(32, 1, 1))
    st = cell_stats_from_capture(tmp_path, "c")
    assert st.n_blocks == 4 and st.grid == (4, 1, 1)
    assert st.total_sector_accesses == 10
    assert st.stats_pass_version == 2
    assert not st.serialized_bytes_exact


def test_cell_stats_mode2_carries_exact_serialized_bytes(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1), mode=2)
    st = cell_stats_from_capture(tmp_path, "c")
    assert st.serialized_bytes_exact
    assert st.total_serialized_bytes == 201


def test_cell_stats_old_stats_pass_version_is_rejected(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1),
                   stats_pass_version=1)
    with pytest.raises(ValueError, match="stats_pass_version"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_atomic_sketch_overflow_is_rejected(tmp_path):
    import json
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1))
    path = tmp_path / "extra_info" / "trace_sampling_capture.json"
    capture = json.loads(path.read_text())
    capture["atomic_sectors_overflow"] = True
    path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="sketch overflow"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_invalid_sector_marker_is_rejected(tmp_path):
    import json
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1))
    path = tmp_path / "extra_info" / "trace_sampling_capture.json"
    capture = json.loads(path.read_text())
    capture["sector_accesses_valid"] = False
    path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="sector_accesses invalid"):
        cell_stats_from_capture(tmp_path, "c")


@pytest.mark.parametrize("field", ["stats_pass_version", "sector_accesses_valid"])
def test_cell_stats_missing_v2_evidence_field_is_rejected(tmp_path, field):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1), omit=[field])
    with pytest.raises(ValueError, match=field):
        cell_stats_from_capture(tmp_path, "c")


@pytest.mark.parametrize("value", [-1, True, 1.0, "1"])
def test_cell_stats_malformed_sector_accesses_is_rejected(tmp_path, value):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1),
                   block_overrides={"sector_accesses": value})
    with pytest.raises(ValueError, match="sector_accesses|integer|nonnegative"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_missing_sector_accesses_is_rejected(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1),
                   block_omit=["sector_accesses"])
    with pytest.raises(ValueError, match="sector_accesses"):
        cell_stats_from_capture(tmp_path, "c")


@pytest.mark.parametrize(("overrides", "message"), [
    ({"records": 0}, "records must be positive"),
    ({"max_warpid_sm": -1}, "max_warpid_sm"),
    ({"records": 1, "atomic_records": 2}, "cannot exceed records"),
])
def test_cell_stats_impossible_producer_row_is_rejected(
        tmp_path, overrides, message):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1),
                   block_overrides=overrides)
    with pytest.raises(ValueError, match=message):
        cell_stats_from_capture(tmp_path, "c")


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "100"])
def test_cell_stats_mode2_invalid_serialized_bytes_is_rejected(tmp_path, value):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1), mode=2,
                   block_overrides={"serialized_bytes": value})
    with pytest.raises(ValueError, match="serialized_bytes|integer|nonnegative"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_empty_blocks_is_rejected(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1), ctas=[])
    with pytest.raises(ValueError, match="non-empty"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_incomplete_all_block_evidence_is_rejected(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(2, 1, 1), block=(32, 1, 1),
                   ctas=[(0, 0, 0)])
    with pytest.raises(ValueError, match="all-block evidence incomplete"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_grid_mismatch_raises(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(4, 1, 1), block=(32, 1, 1), cap_grid=(8, 1, 1))
    with pytest.raises(ValueError, match="grid_dim"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_missing_shape_raises(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(4, 1, 1), block=(32, 1, 1), omit=["grid_dim"])
    with pytest.raises(ValueError, match="missing grid_dim"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_out_of_grid_cta_raises(tmp_path):
    from counting_pass import cell_stats_from_capture
    # cta.x = 4 is outside grid.x = 4.
    _write_capture(tmp_path, grid=(4, 1, 1), block=(32, 1, 1), ctas=[[4, 0, 0]])
    with pytest.raises(ValueError, match="outside grid"):
        cell_stats_from_capture(tmp_path, "c")


# --- strict-integer validation (no int() coercion, Codex R18) ---

def test_cell_stats_float_grid_dim_is_rejected(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(4, 1, 1), block=(32, 1, 1), cap_grid=(4.0, 1, 1))
    with pytest.raises(ValueError, match="integer"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_string_grid_dim_is_rejected(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(4, 1, 1), block=(32, 1, 1), cap_grid=("4", 1, 1))
    with pytest.raises(ValueError, match="integer"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_float_cta_is_rejected(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(4, 1, 1), block=(32, 1, 1), ctas=[[0.9, 0, 0]])
    with pytest.raises(ValueError, match="integer"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_string_cta_is_rejected(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(4, 1, 1), block=(32, 1, 1), ctas=[["0", 0, 0]])
    with pytest.raises(ValueError, match="integer"):
        cell_stats_from_capture(tmp_path, "c")


def test_cell_stats_string_launch_kernel_is_rejected(tmp_path):
    from counting_pass import cell_stats_from_capture
    _write_capture(tmp_path, grid=(4, 1, 1), block=(32, 1, 1),
                   launch=(0, 0, 1), cap_launch=(0, 0, "1"))
    with pytest.raises(ValueError, match="integer"):
        cell_stats_from_capture(tmp_path, "c")


@pytest.mark.skipif(not _FIXTURE.exists(), reason="canary fixture absent")
def test_decode_real_canary_trace():
    stats = analyze_trace_dir(_FIXTURE, cell_id="addr_carry_canary")
    assert stats.grid == (1, 1, 1)
    assert stats.n_blocks == 1
    assert stats.total_records > 0
    # The canary carries admitted atomic + async-copy records post-flip.
    assert stats.op_class_totals.get(ATOMIC, 0) > 0
    assert stats.op_class_totals.get(ASYNC_COPY, 0) > 0
    assert stats.atomic_records > 0
