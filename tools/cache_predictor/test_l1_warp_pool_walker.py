import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.cache_predictor import l1_warp_pool_walker as walker
from tools.cache_predictor import l1_trace_adapter as trace_adapter
from tools.cache_predictor import l1_production_helper as production_helper
from tools.cache_predictor import l1_cache_op_classifier as classifier
from tools.cache_predictor import l1_walker as l1_legacy
from tools.cache_predictor.manifest_join import join_manifest_to_records

from tools.cache_predictor.l1_warp_pool_walker import (
    WalkerConfigError,
    WarpPoolAccessInvalid,
    resolve_latencies,
    resolve_line_capacity,
    walk_warp_pool,
)
from tools.cache_predictor.l1_walker import (
    L1CtaToSmScheduleAssignmentInvalid,
    L1CtaToSmScheduleMissingOccupancy,
    L1CtaToSmScheduleUnknown,
    L1WalkerAccessInvalid,
    L1WalkerCapacityInvalid,
    L1WalkerWaysInvalid,
    apply_carveout_to_capacity,
    block_to_sm_id,
    walk_per_sm_l1,
    walk_per_sm_per_set_l1,
)


def _card(tmp_path, *, capacity_unit="bytes", latency_unit="cycles_per_dependent_load"):
    document = {
        "arch": "sm_86",
        "properties": {
            "l1_capacity_bytes": {
                "kind": "value",
                "unit": capacity_unit,
                "value": 98304,
            },
            "l1_hit_latency_cycles": {
                "kind": "value",
                "unit": latency_unit,
                "value": 33.0,
            },
            "l2_hit_latency_cycles": {
                "kind": "value",
                "unit": latency_unit,
                "value": 243.0,
            },
            "mshr_outstanding_misses": {
                "kind": "value",
                "value": 32,
            },
        },
    }
    path = tmp_path / "card.json"
    path.write_text(json.dumps(document))
    return path


def test_explicit_capacity_unit_must_be_bytes(tmp_path):
    with pytest.raises(WalkerConfigError, match="is not bytes"):
        resolve_line_capacity(_card(tmp_path, capacity_unit="cycles"))


@pytest.mark.parametrize("payload", ["{", "[]"])
def test_card_json_root_errors_stay_typed(tmp_path, payload):
    path = tmp_path / "malformed-card.json"
    path.write_text(payload)
    with pytest.raises(WalkerConfigError, match="(valid JSON|root must be)"):
        resolve_line_capacity(path)


def test_explicit_latency_unit_must_be_cycles(tmp_path):
    with pytest.raises(WalkerConfigError, match="is not cycles_per_dependent_load"):
        resolve_latencies(_card(tmp_path, latency_unit="nanoseconds"))


def test_large_decimal_capacity_is_not_rounded(tmp_path):
    path = _card(tmp_path)
    path.write_text(path.read_text().replace("98304", "9007199254740993.0", 1))
    with pytest.raises(WalkerConfigError, match="9007199254740993"):
        resolve_line_capacity(path)


def test_card_integer_bounds_match_rust_resolvers(tmp_path):
    path = _card(tmp_path)
    raw = path.read_text().replace('"value": 32', '"value": 4294967296', 1)
    path.write_text(raw)
    from tools.cache_predictor.l1_warp_pool_walker import resolve_mshr_count

    with pytest.raises(WalkerConfigError, match="does not fit u32"):
        resolve_mshr_count(path)


def test_huge_exponent_is_rejected_before_integer_materialization(tmp_path):
    path = _card(tmp_path)
    path.write_text(path.read_text().replace("98304", "1e1000000", 1))
    with pytest.raises(WalkerConfigError, match="positive integer"):
        resolve_line_capacity(path)


@pytest.mark.parametrize("token", ["1e-400", "1e400", "1" + "0" * 400])
def test_latency_must_remain_positive_and_finite_after_conversion(tmp_path, token):
    path = _card(tmp_path)
    path.write_text(path.read_text().replace("33.0", token, 1))
    with pytest.raises(WalkerConfigError, match="finite latency"):
        resolve_latencies(path)


def test_representable_subnormal_latency_is_preserved(tmp_path):
    path = _card(tmp_path)
    path.write_text(path.read_text().replace("33.0", "5e-324", 1))
    assert resolve_latencies(path).hit_latency == float.fromhex("0x0.0000000000001p-1022")


def test_card_size_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(walker, "MAX_CARD_BYTES", 1024)
    path = _card(tmp_path)
    exact_limit = path.read_bytes().ljust(walker.MAX_CARD_BYTES, b" ")
    path.write_bytes(exact_limit)
    assert resolve_line_capacity(path).capacity_bytes == 98304

    path.write_bytes(exact_limit + b" " * 4096)
    original_open = Path.open
    consumed = []

    @contextmanager
    def tracked_open(card_path, *args, **kwargs):
        with original_open(card_path, *args, **kwargs) as stream:
            def read(size=-1):
                raw = stream.read(size)
                consumed.append(len(raw))
                return raw
            yield SimpleNamespace(read=read)

    monkeypatch.setattr(Path, "open", tracked_open)
    with pytest.raises(WalkerConfigError, match="exceeds"):
        resolve_line_capacity(path)
    assert consumed and sum(consumed) <= walker.MAX_CARD_BYTES + 1


def test_card_path_rejects_leaf_and_parent_symlinks(tmp_path):
    target = _card(tmp_path)
    leaf = tmp_path / "leaf.json"
    leaf.symlink_to(target)
    with pytest.raises(WalkerConfigError, match="contains a symlink"):
        resolve_line_capacity(leaf)
    parent = tmp_path / "parent"
    parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(WalkerConfigError, match="contains a symlink"):
        resolve_line_capacity(parent / target.name)


def test_duplicate_warp_keys_cannot_drop_streams():
    with pytest.raises(WalkerConfigError, match="unique"):
        walk_warp_pool(
            [("warp", [(0, True)]), ("warp", [(1, True)])],
            line_capacity=8, hit_latency=1.0, miss_latency=4.0, mshr_count=1,
        )


@pytest.mark.parametrize("warmup_launches", [0, 1])
def test_one_shot_outer_and_stream_iterators_survive_validation(warmup_launches):
    workloads = (
        (key, (access for access in [(0, True), (0, True)]))
        for key in ["warp"]
    )
    result = walk_warp_pool(
        workloads,
        line_capacity=8,
        hit_latency=1.0,
        miss_latency=4.0,
        mshr_count=1,
        warmup_launches=warmup_launches,
    )
    if warmup_launches == 0:
        assert (result.hit_count, result.miss_count) == (1, 1)
    else:
        assert (result.hit_count, result.miss_count) == (2, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warps_per_block": True},
        {"blocks_per_sm_limit": 1.5},
        {"occupancy_warp_limit": 0},
    ],
)
def test_co_resident_geometry_rejects_coercible_values(kwargs):
    values = {
        "warps_per_block": 2,
        "blocks_per_sm_limit": 2,
        "occupancy_warp_limit": 48,
    }
    values.update(kwargs)
    with pytest.raises(WalkerConfigError, match="positive integer"):
        walker.co_resident_warp_limit(**values)


def test_non_tuple_access_has_typed_error():
    with pytest.raises(WarpPoolAccessInvalid, match="tuple/list"):
        walk_warp_pool(
            [("warp", [123])], line_capacity=8,
            hit_latency=1.0, miss_latency=4.0, mshr_count=1,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mshr_count": 0}, "mshr_count"),
        ({"co_resident_warps": 0}, "co_resident_warps"),
        ({"hit_latency": float("nan")}, "latencies"),
        ({"hit_latency": 5.0, "miss_latency": 4.0}, "latencies"),
    ],
)
def test_invalid_pool_parameters_fail_closed(kwargs, message):
    base = {
        "line_capacity": 8,
        "hit_latency": 1.0,
        "miss_latency": 4.0,
        "mshr_count": 1,
    }
    base.update(kwargs)
    with pytest.raises(WalkerConfigError, match=message):
        walk_warp_pool([("warp", [(0, True)])], **base)


def test_legacy_walker_rejects_non_integral_geometry():
    with pytest.raises(L1WalkerCapacityInvalid, match="capacity_bytes"):
        walk_per_sm_l1([], sm_count=1, capacity_bytes=128.5, schedule_policy="round_robin_homogeneous")
    with pytest.raises(L1WalkerWaysInvalid, match="ways"):
        walk_per_sm_per_set_l1([], sm_count=1, capacity_bytes=128, ways=1.5, schedule_policy="round_robin_homogeneous")


def test_legacy_walkers_reject_zero_sm_count_on_empty_input():
    with pytest.raises(ValueError, match="sm_count"):
        walk_per_sm_l1([], sm_count=0, capacity_bytes=128, schedule_policy="round_robin_homogeneous")
    with pytest.raises(ValueError, match="sm_count"):
        walk_per_sm_per_set_l1([], sm_count=0, capacity_bytes=128, ways=1, schedule_policy="round_robin_homogeneous")


def test_occupancy_assignment_rejects_coercible_non_integer_ids():
    with pytest.raises(L1CtaToSmScheduleAssignmentInvalid, match="non-integer"):
        block_to_sm_id(0, sm_count=1, schedule_policy="occupancy_derived", occupancy_derived_assignment=[True])


def test_empty_legacy_walkers_validate_schedule_policy():
    with pytest.raises(L1CtaToSmScheduleUnknown):
        walk_per_sm_l1([], sm_count=1, capacity_bytes=128, schedule_policy="unknown")
    with pytest.raises(L1CtaToSmScheduleMissingOccupancy):
        walk_per_sm_per_set_l1([], sm_count=1, capacity_bytes=128, ways=1, schedule_policy="occupancy_derived")


@pytest.mark.parametrize("policy", ["round_robin_homogeneous", "occupancy_derived"])
@pytest.mark.parametrize("index", [-1, True, 0.5])
def test_block_mapping_rejects_invalid_indices(policy, index):
    with pytest.raises(L1CtaToSmScheduleAssignmentInvalid, match="block_index"):
        block_to_sm_id(
            index, sm_count=4, schedule_policy=policy,
            occupancy_derived_assignment=[0, 2],
        )


@pytest.mark.parametrize("walker_kind", ["per_sm", "warp_pool"])
def test_sector_id_must_belong_to_declared_line(walker_kind):
    if walker_kind == "per_sm":
        with pytest.raises(L1WalkerAccessInvalid, match="sector_id"):
            walk_per_sm_l1(
                [[(0, 4, True)]], sm_count=1, capacity_bytes=128,
                schedule_policy="round_robin_homogeneous",
            )
    else:
        with pytest.raises(walker.WarpPoolAccessInvalid, match="sector_id"):
            walker.walk_warp_pool(
                [("warp", [(0, 4, True)])], line_capacity=1,
                hit_latency=1.0, miss_latency=4.0, mshr_count=1,
            )


@pytest.mark.parametrize("raw_access", [(1.5, True), (0, "false"), (-1, True)])
def test_per_sm_walker_rejects_coercible_access_fields(raw_access):
    with pytest.raises(L1WalkerAccessInvalid, match="access fields"):
        walk_per_sm_l1(
            [[raw_access]], sm_count=1, capacity_bytes=128,
            schedule_policy="round_robin_homogeneous",
        )


@pytest.mark.parametrize("raw_access", [(1.5, True), (0, "false"), (-1, True)])
def test_warp_pool_walker_rejects_coercible_access_fields(raw_access):
    with pytest.raises(walker.WarpPoolAccessInvalid, match="access fields"):
        walker.walk_warp_pool(
            [("warp", [raw_access])], line_capacity=1,
            hit_latency=1.0, miss_latency=4.0, mshr_count=1,
        )


@pytest.mark.parametrize("carveout_percent", [True, 50.9, "50"])
def test_carveout_percent_rejects_coercion(carveout_percent):
    with pytest.raises(ValueError, match="carveout_percent"):
        apply_carveout_to_capacity(
            unified_budget_bytes=160 * 1024,
            static_shared_bytes=0,
            dynamic_shared_bytes=0,
            carveout_source="cuda_func_attribute_carveout_percent",
            carveout_max_l1_bytes=128 * 1024,
            default_l1_bytes=16 * 1024,
            carveout_percent=carveout_percent,
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("unified_budget_bytes", -1),
        ("static_shared_bytes", -1),
        ("dynamic_shared_bytes", 1.5),
        ("carveout_max_l1_bytes", "131072"),
        ("default_l1_bytes", True),
    ],
)
def test_carveout_byte_inputs_reject_coercion(field, value):
    kwargs = {
        "unified_budget_bytes": 160 * 1024,
        "static_shared_bytes": 0,
        "dynamic_shared_bytes": 0,
        "carveout_source": "default",
        "carveout_max_l1_bytes": 128 * 1024,
        "default_l1_bytes": 16 * 1024,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        apply_carveout_to_capacity(**kwargs)


@pytest.mark.parametrize(
    "field, value",
    [
        ("block_index", 0.5),
        ("block_index", None),
        ("warp_index", True),
        ("sm_id", -1),
        ("byte_addresses", [1.5]),
        ("byte_addresses", None),
        ("pc", 16.5),
        ("pc", True),
        ("pc", "not_hex"),
    ],
)
def test_trace_adapter_rejects_coercible_fields(field, value):
    record = {
        "block_index": 0,
        "pc": 16,
        "byte_addresses": [32],
        "data_width": 4,
        "warp_index": 0,
        "sm_id": 0,
    }
    record[field] = value
    manifest = {
        "sass_provenance_status": "matched",
        "per_pc_instruction_cache_op": [
            {"pc": "0x10", "instruction_cache_op": "LDG_E_CA", "l1_eligible": True},
        ],
    }
    for raw in (record, trace_adapter.TraceRecord(**record)):
        for build in (trace_adapter.build_per_block_l1_accesses,
                      trace_adapter.build_per_warp_l1_accesses):
            with pytest.raises(trace_adapter.L1TraceAdapterRecordInvalid, match=field):
                build([raw], manifest, sm_count=1, grid_dim=1, block_dim=32)


def test_trace_adapter_preserves_integer_pc_and_legacy_width():
    record = trace_adapter._materialize_record({
        "block_index": 0,
        "pc": 16,
        "byte_addresses": [32],
        "data_width": None,
    })
    assert record.pc == "0x10"
    assert record.byte_addresses == (32,)
    assert record.data_width is None


def test_trace_adapter_preserves_width_diagnostics():
    for width in ("4", 4.5, True, 0, -1):
        with pytest.raises(trace_adapter.L1TraceAdapterDataWidthInvalid) as failure:
            trace_adapter._materialize_record({
                "block_index": 0, "pc": 16, "byte_addresses": [32], "data_width": width,
            })
        assert failure.value.to_evidence()["data_width"] == width


def test_trace_adapter_public_builders_preserve_sector_and_pc_identity():
    manifest = {
        "sass_provenance_status": "matched",
        "per_pc_instruction_cache_op": [
            {"pc": "0x10", "instruction_cache_op": "LDG_E_CA", "l1_eligible": True},
        ],
    }
    for pc in (16, "10", "  0X0010  "):
        raw = {"block_index": 0, "pc": pc, "byte_addresses": [31, 31, 64],
               "data_width": 4, "warp_index": 0}
        for record in (raw, trace_adapter.TraceRecord(**raw)):
            expected = [(0, 0, True), (0, 1, True), (0, 2, True)]
            assert trace_adapter.build_per_block_l1_accesses(
                [record], manifest, sm_count=1, grid_dim=1, block_dim=32,
                require_data_width=True,
            ) == [expected]
            assert trace_adapter.build_per_warp_l1_accesses(
                [record], manifest, sm_count=1, grid_dim=1, block_dim=32,
                require_data_width=True,
            ) == [((0, 0), expected)]


def test_trace_adapter_rejects_duplicate_manifest_pc():
    manifest = {
        "sass_provenance_status": "matched",
        "per_pc_instruction_cache_op": [
            {"pc": "0x10", "instruction_cache_op": "LDG_E_CA", "l1_eligible": True},
            {"pc": "0x0010", "instruction_cache_op": "LDG_E_CG", "l1_eligible": False},
        ],
    }
    with pytest.raises(trace_adapter.L1TraceAdapterEligibilityDefect, match="duplicate PC"):
        trace_adapter.build_per_block_l1_accesses(
            [{"block_index": 0, "pc": 16, "byte_addresses": [32], "data_width": 4}],
            manifest,
            sm_count=1,
            grid_dim=1,
            block_dim=32,
            require_data_width=True,
        )


@pytest.mark.parametrize("raw", [
    {"block_index": 0.5, "sm_id": 0},
    {"block_index": 0, "sm_id": "0"},
    {"block_index": True, "sm_id": 0},
    {"block_index": 0, "sm_id": -1},
])
def test_cta_partition_rejects_coercible_identity_fields(raw):
    with pytest.raises(trace_adapter.L1TraceAdapterRecordInvalid, match="(block_index|sm_id)"):
        trace_adapter.derive_cta_to_sm_assignment([raw], grid_dim=1, sm_count=1)


@pytest.mark.parametrize("field, value", [
    ("sm_count", True), ("sm_count", 1.5),
    ("grid_dim", True), ("grid_dim", 1.5),
    ("block_dim", True), ("block_dim", 32.5),
])
def test_trace_adapter_rejects_coercible_geometry(field, value):
    kwargs = {"sm_count": 1, "grid_dim": 1, "block_dim": 32}
    kwargs[field] = value
    with pytest.raises(trace_adapter.L1TraceAdapterGridBlockMissing):
        trace_adapter.build_per_block_l1_accesses(
            [], {"sass_provenance_status": "matched"}, **kwargs
        )


@pytest.mark.parametrize("warmup_launches", [True, 1.5, -1])
def test_l1_miss_stream_rejects_coercible_warmup_count(warmup_launches):
    with pytest.raises(ValueError, match="warmup_launches"):
        l1_legacy.l1_miss_stream_per_sm(
            [[(0, True)]],
            sm_count=1,
            capacity_bytes=128,
            schedule_policy="round_robin_homogeneous",
            warmup_launches=warmup_launches,
        )


def test_production_proto_walker_preserves_optional_sm_id(monkeypatch, tmp_path):
    """The real CTA partition must survive proto -> TraceRecord decoding."""

    class _BlockId:
        x = y = z = 0

    class _Address:
        def __init__(self, data_width):
            self.data_width = data_width

    class _Instruction:
        # Keep the manifest key function-relative while exercising a widened
        # vpc.  The production join must use pc, not the optional high bits.
        pc = 0x20
        vpc = 0x100000020
        active_mask = predicate_mask = 1
        data_width = 4
        sm_id = 3
        addresses = [_Address(4), _Address(16), _Address(0)]

        def HasField(self, name):
            return name in {
                "memory_op_class", "active_mask", "predicate_mask",
                "data_width", "warpid_tb", "sm_id",
            }

    class _Warp:
        instructions = [_Instruction()]

    class _Threadblock:
        block_id = _BlockId()
        warps = {0: _Warp()}

        def HasField(self, name):
            return name == "block_id"

        def ParseFromString(self, raw):
            assert raw == b"synthetic-threadblock"

    class _Bindings:
        threadblock = _Threadblock

    monkeypatch.setattr(
        production_helper, "preflight_threadblock_protobuf", lambda raw: None
    )
    monkeypatch.setattr(
        production_helper,
        "_reconstruct_lane_addresses",
        lambda address, and_mask, contract: [0x1000],
    )

    records = production_helper._walk_threadblock_protos(
        _Bindings(),
        [tmp_path / "0.pb"],
        object(),
        grid_x=1,
        grid_y=1,
        grid_z=1,
        grid_dim=1,
        retained_threadblock_pb_payloads=[b"synthetic-threadblock"],
    )

    assert len(records) == 3
    assert [record.pc for record in records] == ["0x20", "0x20", "0x20"]
    assert [record.data_width for record in records] == [4, 16, 4]
    assert [record.sm_id for record in records] == [3, 3, 3]


def test_classifier_selects_requested_arch_and_exact_first_symbol(monkeypatch, tmp_path):
    calls = []

    class _Completed:
        returncode = 0
        stderr = ""
        stdout = """\
Fatbin elf code:
arch = sm_120
\tFunction : target_helper
        /*0000*/ LDG.E.CG R0, [R2];
\tFunction : target
        /*0010*/ LDG.E.CA R0, [R2];
\tFunction : target
        /*0020*/ LDG.E.CG R0, [R2];
"""

    def _run(args, **kwargs):
        calls.append((args, kwargs))
        return _Completed()

    monkeypatch.setattr(classifier.subprocess, "run", _run)
    binary = tmp_path / "fatbin"
    binary.write_bytes(b"placeholder")
    manifest = classifier.classify_binary(
        binary,
        kernel_unmangled_name="target",
        arch_token="sm_86",
    )

    assert calls[0][0] == [
        "cuobjdump", "--dump-sass", "--gpu-architecture", "sm_86",
        str(binary),
    ]
    entries = manifest["per_pc_instruction_cache_op"]
    assert [(entry["pc"], entry["instruction_cache_op"]) for entry in entries] == [
        ("0x10", "LDG_E_CA"),
    ]


def test_classifier_rejects_duplicate_load_class_pc():
    with pytest.raises(classifier.SassManifestDuplicatePcError, match="duplicate load-class PC"):
        classifier.classify_sass_text(
            """\
Function : target
    /*0010*/ LDG.E.CA R0, [R2];
    /*0010*/ LDG.E.CG R0, [R2];
""",
            kernel_unmangled_name="target",
            arch_token="sm_86",
            binary_source="canned_fixture",
        )


def test_manifest_join_rejects_duplicate_normalized_pc():
    with pytest.raises(ValueError, match="duplicate normalized PC 0x10"):
        join_manifest_to_records(
            {
                "per_pc_instruction_cache_op": [
                    {"pc": "0x10", "instruction_cache_op": "LDG_E_CA", "l1_eligible": True},
                    {"pc": "0010", "instruction_cache_op": "LDG_E_CG", "l1_eligible": False},
                ],
            },
            [],
        )


@pytest.mark.parametrize("pc", [True, -1, "0x", "0xgg", "not_hex"])
def test_manifest_join_rejects_malformed_pc_identity(pc):
    with pytest.raises((TypeError, ValueError), match="PC"):
        join_manifest_to_records(
            {
                "per_pc_instruction_cache_op": [
                    {"pc": pc, "instruction_cache_op": "LDG_E_CA", "l1_eligible": True},
                ],
            },
            [],
        )


def test_classifier_keeps_global_stores_in_the_l2_bypass_manifest():
    assert classifier._match_opcode_token(
        "STG.E.128 [R8.64], R4", arch_token="sm_86"
    ) == ("OTHER", "")
    manifest = classifier.classify_sass_text(
        """\
Function : kernel
    /*04a0*/ STG.E.128 [R8.64], R4 ;
""",
        kernel_unmangled_name="kernel",
        arch_token="sm_86",
        binary_source="canned_fixture",
    )
    assert manifest["sass_provenance_status"] == "matched"
    assert manifest["per_pc_instruction_cache_op"] == [{
        "pc": "0x4a0",
        "instruction_cache_op": "OTHER",
        "l1_eligible": False,
    }]
