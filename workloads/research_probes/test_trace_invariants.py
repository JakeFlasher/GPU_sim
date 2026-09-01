from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings
from workloads.research_probes import trace_invariants


def test_operand_lane_mask_excludes_async_zero_fill_source_lane() -> None:
    bindings = load_dynamic_trace_bindings()
    instruction = bindings.instruction_pb2.instruction()
    instruction.active_mask = 0b11
    instruction.predicate_mask = 0b11
    instruction.data_width = 16
    address = instruction.addresses.add()
    address.data_width = 16
    address.compression_format = 0
    address.lane_mask = 0b01
    address.addrs.extend([0x1000, *([-1] * 31)])

    assert trace_invariants._operand_sectors(instruction, [0, 1], 0) == 1
