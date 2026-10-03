from pathlib import Path
import json

import pytest
from extract_sm86_tail_facts import extract_tail_facts, threadblock_pb2, trace_pb2

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, functions=("a",), grids=(2,)):
    root = trace_pb2.Trace()
    stream = root.gpu_device[0].streams[0]
    for kid, (name, blocks) in enumerate(zip(functions, grids), 1):
        kernel = stream.kernels.add(id=kid, function_unique_id=kid, name=name + "___0")
        kernel.grid_dim.x = blocks
        kernel.grid_dim.y = kernel.grid_dim.z = 1
    (tmp_path / "dynamic_trace.pb").write_bytes(root.SerializeToString())
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"functions": {name: [{"pc": 0, "opcode_text": "STG.E [R2], R0"}] for name in functions}}))
    return control


def payload(tmp_path, kid, cta, *, format=0, address=128, opcode_class=1, access=2, mask=3, order=None):
    block = threadblock_pb2.threadblock()
    block.block_id.x = cta
    inst = block.warps[0].instructions.add(
        pc=0, function_unique_id=kid, active_mask=mask, predicate_mask=mask,
        sm_id=cta, memory_op_class=opcode_class, memory_op_access_kind=access,
    )
    if order is not None:
        inst.instrumentation_order = order
    addr = inst.addresses.add(data_width=4, address_space=1, allocation_id=1, compression_format=format)
    if format == 0:
        addr.addrs.extend([address, address + 4] + [-1] * 30)
    elif format == 1:
        addr.base_address, addr.stride = address, 4
    else:
        addr.base_address = address
        addr.addrs.append(4)
    path = tmp_path / f"threadblocks/device_0/stream_0/kernel_{kid}/d_{cta}.pb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(block.SerializeToString())
    return path


@pytest.mark.parametrize("format", [0, 1, 2])
def test_decodes_each_compression_format_and_width(tmp_path, format):
    control = fixture(tmp_path)
    # The four-byte span starting at 126 straddles two 128-byte lines.
    payload(tmp_path, 1, 0, format=format, address=126)
    payload(tmp_path, 1, 1, format=format, address=126)
    report = extract_tail_facts(tmp_path, control)
    assert report["facts"]["cross_sm_line_topology"] == 2
    assert report["coverage"]["kind"] == "full_block_roster"
    assert report["facts"]["global_event_order_observed"] is False
    assert report["facts"]["instrumentation_order_strictly_unique"] is None


def test_function_pc_join_red_and_launch_scoping(tmp_path):
    control = fixture(tmp_path, ("a", "b"), (1, 2))
    control.write_text(json.dumps({"functions": {
        "a": [{"pc": 0, "opcode_text": "RED.E.ADD [R2], R0"}],
        "b": [{"pc": 0, "opcode_text": "DEPBAR.LE SB1, 0x0"}],
    }}))
    payload(tmp_path, 1, 0, access=3, opcode_class=6, order=1)
    payload(tmp_path, 2, 1, access=1, order=1)
    report = extract_tail_facts(tmp_path, control)
    assert report["facts"]["reductions"] == 1
    assert report["facts"]["atomics"] == 0
    assert report["facts"]["protocol_record_counts"]["async_wait_records"] == 0
    assert report["facts"]["cross_sm_line_topology"] == 0  # not cross-launch sharing
    assert report["facts"]["instrumentation_order_strictly_unique"] is True
    assert report["coverage"]["kind"] == "partial_block_roster"


def test_incomplete_masks_do_not_invent_topology(tmp_path):
    control = fixture(tmp_path)
    p = payload(tmp_path, 1, 0)
    block = threadblock_pb2.threadblock.FromString(p.read_bytes())
    block.warps[0].instructions[0].ClearField("predicate_mask")
    p.write_bytes(block.SerializeToString())
    report = extract_tail_facts(tmp_path, control)
    assert report["facts"]["predicate_mask_missing_records"] == 1
    assert report["facts"]["address_operands_unknown"] == 1
    assert report["facts"]["stores"] == 0


def test_wrong_function_cannot_fall_back_to_same_pc(tmp_path):
    control = fixture(tmp_path)
    payload(tmp_path, 1, 0)
    control.write_text(json.dumps({"functions": {"wrong": [{"pc": 0, "opcode_text": "MEMBAR.SC.GPU"}]}}))
    with pytest.raises(ValueError, match="no function"):
        extract_tail_facts(tmp_path, control)


@pytest.mark.parametrize("trace,control,expected", [
    ("c1r_message_device_20261003/trace", "c1r_message_device_20261003/control.json", (752, 32, 0)),
    ("c1r_async_matched_20261003/trace", "c1r_async_matched_20261003/control.json", (9400, 0, 4)),
])
def test_live_trace_facts(trace, control, expected):
    root = ROOT / "results/alpha"
    if not (root / trace).is_dir():
        pytest.skip("local live capture not installed")
    report = extract_tail_facts(root / trace, root / control)
    facts = report["facts"]
    assert (facts["records"], facts["cross_sm_line_topology"], facts["reductions"]) == expected
    assert facts["owner_read_from_observed"] is False
    assert facts["hardware_completion_token_observed"] is False
    assert facts["static_opcode_records"] == facts["records"]


def test_async_wait_prefix_preserves_empty_groups_and_thread_masks():
    from extract_sm86_tail_facts import _async_protocol
    b = threadblock_pb2.threadblock()
    ops = {(1, 0): 'LDGSTS.E.128 [R2], [R4.64]',
           (1, 16): 'LDGDEPBAR', (1, 32): 'DEPBAR.LE SB0, 0x1'}
    for pc, mask in [(0, 3), (16, 3), (16, 1), (32, 3)]:
        b.warps[0].instructions.add(function_unique_id=1, pc=pc, active_mask=3, predicate_mask=mask)
    facts = _async_protocol(b.warps[0].instructions, ops)
    assert facts['committed_lane_groups'] == 3
    assert facts['empty_lane_groups'] == 1
    assert facts['waited_lane_copies'] == 1  # lane 0 waits the old copy; lane 1 retains it
    assert facts['terminal_unwaited_lane_copies'] == 1
