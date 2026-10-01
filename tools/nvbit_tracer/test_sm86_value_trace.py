#!/usr/bin/env python3
import unittest
import hashlib
import json
import tempfile
import random
from pathlib import Path

from sm86_value_trace import TraceRefusal, execute, _load_manifest


SASS = b"""        /*0000*/ S2R R1, SR_TID.X ;
        /*0010*/ IADD3 R2, R1, 0x2, RZ ;
        /*0020*/ MOV R3, 0x100 ;
        /*0030*/ STG.E [R3.64], R2 ;
        /*0040*/ EXIT ;
"""

BRANCH_SASS = b"""        /*0000*/ S2R R1, SR_TID.X ;
        /*0010*/ ISETP.GE.U32.AND P0, PT, R1, 0x4, PT ;
        /*0020*/ @P0 IADD3 R2, R1, 0xa, RZ ;
        /*0030*/ @!P0 IADD3 R2, R1, 0x14, RZ ;
        /*0040*/ MOV R3, 0x100 ;
        /*0050*/ STG.E [R3.64], R2 ;
        /*0060*/ EXIT ;
"""

LOAD_SASS = b"""        /*0000*/ S2R R1, SR_TID.X ;
        /*0010*/ IADD3 R3, R1, 0x100, RZ ;
        /*0020*/ LDG.E R2, [R3.64] ;
        /*0030*/ MOV R4, 0x200 ;
        /*0040*/ STG.E [R4.64], R2 ;
        /*0050*/ EXIT ;
"""

BRA_SASS = b"""        /*0000*/ S2R R1, SR_TID.X ;
        /*0010*/ ISETP.GE.U32.AND P0, PT, R1, 0x4, PT ;
        /*0020*/ @P0 BRA 0x50 ;
        /*0030*/ IADD3 R2, R1, 0x14, RZ ;
        /*0040*/ BRA 0x60 ;
        /*0050*/ IADD3 R2, R1, 0xa, RZ ;
        /*0060*/ MOV R3, 0x100 ;
        /*0070*/ STG.E [R3.64], R2 ;
        /*0080*/ EXIT ;
"""

LOP3_SASS = b"""        /*0000*/ S2R R1, SR_TID.X ;
        /*0010*/ LOP3.LUT R2, R1, 0x3, RZ, 0x96, !PT ;
        /*0020*/ MOV R3, 0x100 ;
        /*0030*/ STG.E [R3.64], R2 ;
        /*0040*/ EXIT ;
"""

LOP3_PRED_SASS = b"""        /*0000*/ S2R R5, SR_TID.X ;
        /*0010*/ LOP3.LUT P0, R0, R5, 0x3, RZ, 0xc0, !PT ;
        /*0020*/ @P0 MOV R1, 0xaa ;
        /*0030*/ @!P0 MOV R1, 0x55 ;
        /*0040*/ EXIT ;
"""

SHF_SASS = b"""        /*0000*/ SHF.L.U32 R4, R0, R1, R2 ;
        /*0010*/ SHF.R.U32.HI R5, R0, R1, R2 ;
        /*0020*/ EXIT ;
"""

IMAD_SHL_SASS = b"""        /*0000*/ IMAD.SHL.U32 R4, R0, 0x10, R1 ;
        /*0010*/ EXIT ;
"""

IMAD_SHL_PTXAS_SASS = b"""        /*00f0*/ IMAD.SHL.U32 R3, R0, 0x10, RZ ;
        /*0100*/ EXIT ;
"""

IMAD_IADD_SASS = b"""        /*0000*/ IMAD.IADD R4, R0, 0x1, R1 ;
        /*0010*/ EXIT ;
"""


class Sm86ValueTraceTests(unittest.TestCase):
    def test_lane_value_changes_dynamic_store(self):
        lane7 = execute(SASS, lane_id=7)
        lane8 = execute(SASS, lane_id=8)
        self.assertEqual(lane7["schema"], "sm86_value_trace_v1")
        self.assertEqual(lane7["memory"]["256"], 9)
        self.assertEqual(lane8["memory"]["256"], 10)
        self.assertNotEqual(lane7["records"], lane8["records"])
        self.assertTrue(lane7["claims"]["dynamic_value_trace"])
        self.assertFalse(lane7["claims"]["static_sass_regeneration"])

    def test_imad_shl_immediate_multiplier_regenerates_index_value(self):
        result = execute(
            IMAD_SHL_SASS,
            lane_id=0,
            initial_registers={0: 0x10000001, 1: 7},
            max_steps=2,
        )
        self.assertEqual(result["records"][0]["registers_after"]["R4"], 0x10 + 7)

    def test_imad_shl_accepts_exact_ptxas_emitted_rz_form(self):
        result = execute(
            IMAD_SHL_PTXAS_SASS,
            lane_id=0,
            initial_registers={0: 0x12345678},
            max_steps=2,
        )
        self.assertEqual(result["records"][0]["registers_after"]["R3"], 0x23456780)

    def test_imad_shl_randomized_against_integer_oracle(self):
        rng = random.Random(0x1AD86)
        for _ in range(64):
            value = rng.getrandbits(32)
            addend = rng.getrandbits(32)
            multiplier = 1 << rng.randrange(0, 16)
            sass = f"/*0000*/ IMAD.SHL.U32 R4, R0, {multiplier:#x}, R1 ;\n/*0010*/ EXIT ;\n".encode()
            result = execute(
                sass,
                lane_id=0,
                initial_registers={0: value, 1: addend},
                max_steps=2,
            )
            with self.subTest(value=value, addend=addend, multiplier=multiplier):
                self.assertEqual(
                    result["records"][0]["registers_after"]["R4"],
                    (value * multiplier + addend) & 0xFFFFFFFF,
                )

    def test_imad_shl_refuses_non_immediate_or_zero_multiplier(self):
        for operands in [
            "R4, R0, R2, R1",
            "R4, R0, 0, R1",
            "R4, R0, -1, R1",
            "R4, R0, 0x3, R1",
        ]:
            with self.subTest(operands=operands):
                sass = f"/*0000*/ IMAD.SHL.U32 {operands} ;\n/*0010*/ EXIT ;\n".encode()
                with self.assertRaises(TraceRefusal) as raised:
                    execute(sass, lane_id=0, initial_registers={0: 1, 1: 2, 2: 4}, max_steps=2)
                self.assertEqual(raised.exception.code, "imad_shl_multiplier_unsupported")

    def test_imad_iadd_multiplier_one_regenerates_addition(self):
        result = execute(
            IMAD_IADD_SASS,
            lane_id=0,
            initial_registers={0: 0xFFFFFFFF, 1: 2},
            max_steps=2,
        )
        self.assertEqual(result["records"][0]["registers_after"]["R4"], 1)

    def test_imad_iadd_randomized_against_addition_oracle(self):
        rng = random.Random(0x1ADD86)
        for _ in range(64):
            left = rng.getrandbits(32)
            right = rng.getrandbits(32)
            result = execute(
                IMAD_IADD_SASS,
                lane_id=0,
                initial_registers={0: left, 1: right},
                max_steps=2,
            )
            with self.subTest(left=left, right=right):
                self.assertEqual(
                    result["records"][0]["registers_after"]["R4"],
                    (left + right) & 0xFFFFFFFF,
                )

    def test_imad_iadd_refuses_other_multiplier_forms(self):
        for multiplier in ["R2", "0x2"]:
            with self.subTest(multiplier=multiplier):
                sass = f"/*0000*/ IMAD.IADD R4, R0, {multiplier}, R1 ;\n/*0010*/ EXIT ;\n".encode()
                with self.assertRaises(TraceRefusal) as raised:
                    execute(sass, lane_id=0, initial_registers={0: 1, 1: 2, 2: 3}, max_steps=2)
                self.assertEqual(raised.exception.code, "imad_iadd_multiplier_unsupported")

    def test_unsupported_opcode_refuses(self):
        with self.assertRaises(TraceRefusal) as raised:
            execute(b"/*0000*/ CALL 0x0 ;\n/*0010*/ EXIT ;\n", lane_id=0)
        self.assertEqual(raised.exception.code, "opcode_unsupported")

    def test_missing_register_state_refuses_instead_of_defaulting_zero(self):
        with self.assertRaises(TraceRefusal) as raised:
            execute(
                b"/*0000*/ IADD3 R0, R7, 1, RZ ;\n/*0010*/ EXIT ;\n",
                lane_id=0,
            )
        self.assertEqual(raised.exception.code, "register_value_missing")

    def test_predicate_path_changes_dynamic_store(self):
        low = execute(BRANCH_SASS, lane_id=3)
        high = execute(BRANCH_SASS, lane_id=5)
        self.assertEqual(low["memory"]["256"], 23)
        self.assertEqual(high["memory"]["256"], 15)
        self.assertNotEqual(low["records"], high["records"])
        self.assertIn("P0", low["records"][1]["predicates_after"])

    def test_initialized_load_return_and_input_varying_address(self):
        result = execute(LOAD_SASS, lane_id=2, initial_memory={258: 77})
        self.assertEqual(result["memory"]["512"], 77)
        self.assertEqual(result["records"][2]["loads"], [{"address": 258, "value_u32": 77}])
        with self.assertRaises(TraceRefusal) as raised:
            execute(LOAD_SASS, lane_id=3, initial_memory={258: 77})
        self.assertEqual(raised.exception.code, "memory_value_missing")

    def test_predicated_numeric_branch_changes_path(self):
        low = execute(BRA_SASS, lane_id=3)
        high = execute(BRA_SASS, lane_id=5)
        self.assertEqual(low["memory"]["256"], 23)
        self.assertEqual(high["memory"]["256"], 15)
        self.assertGreater(len(low["records"]), 0)

    def test_lop3_lut_regenerates_value_dependent_bitwise_result(self):
        lane0 = execute(LOP3_SASS, lane_id=0)
        lane1 = execute(LOP3_SASS, lane_id=1)
        self.assertEqual(lane0["memory"]["256"], 3)
        self.assertEqual(lane1["memory"]["256"], 2)
        self.assertNotEqual(lane0["records"], lane1["records"])

    def test_lop3_lut_rejects_wide_truth_tables(self):
        sass = LOP3_SASS.replace(b"0x96", b"0x100")
        with self.assertRaises(TraceRefusal) as raised:
            execute(sass, lane_id=0)
        self.assertEqual(raised.exception.code, "lut_unsupported")

    def test_lop3_predicate_and_register_destination_regenerate_path(self):
        low = execute(LOP3_PRED_SASS, lane_id=0)
        high = execute(LOP3_PRED_SASS, lane_id=2)
        self.assertEqual(low["records"][3]["registers_after"]["R1"], 0x55)
        self.assertEqual(high["records"][2]["registers_after"]["R1"], 0xAA)
        self.assertEqual(high["records"][1]["predicates_after"]["P0"], True)

    def test_shf_funnel_shift_matches_documented_concat_semantics(self):
        result = execute(
            SHF_SASS,
            lane_id=0,
            initial_registers={
                0: 0x11223344,
                1: 0x55667788,
                2: 8,
            },
            max_steps=3,
        )
        self.assertEqual(result["records"][0]["registers_after"]["R4"], 0x66778811)
        self.assertEqual(result["records"][1]["registers_after"]["R5"], 0x88112233)

    def test_shf_randomized_against_independent_32bit_oracle(self):
        rng = random.Random(0x865F)
        for _ in range(64):
            a = rng.getrandbits(32)
            b = rng.getrandbits(32)
            shift = rng.randrange(0, 128)
            n = shift & 0x1F
            left = ((b << n) | (a >> (32 - n) if n else 0)) & 0xFFFFFFFF
            right = (((b << (32 - n)) & 0xFFFFFFFF) if n else 0) | (a >> n)
            sass = b"/*0000*/ SHF.L.U32 R4, R0, R1, R2 ;\n/*0010*/ SHF.R.U32.HI R5, R0, R1, R2 ;\n/*0020*/ EXIT ;\n"
            result = execute(
                sass,
                lane_id=0,
                initial_registers={0: a, 1: b, 2: shift},
                max_steps=3,
            )
            with self.subTest(a=a, b=b, shift=shift):
                self.assertEqual(result["records"][0]["registers_after"]["R4"], left)
                self.assertEqual(result["records"][1]["registers_after"]["R5"], right)

    def test_shf_wraps_shift_amount_and_rejects_unmodeled_forms(self):
        sass = b"/*0000*/ SHF.L.U32 R4, R0, R1, R2 ;\n/*0010*/ EXIT ;\n"
        wrapped = execute(
            sass,
            lane_id=0,
            initial_registers={0: 1, 1: 2, 2: 32},
            max_steps=2,
        )
        self.assertEqual(wrapped["records"][0]["registers_after"]["R4"], 2)
        for operands, code in [
            ("R4, R0.HI, R1, R2", "shf_operand_unsupported"),
            ("R4, R0, UR1, R2", "shf_operand_unsupported"),
            ("R4, R0, R1, R2, R3", "shf_form_unsupported"),
        ]:
            with self.subTest(operands=operands):
                bad = f"/*0000*/ SHF.L.U32 {operands} ;\n/*0010*/ EXIT ;\n".encode()
                with self.assertRaises(TraceRefusal) as raised:
                    execute(bad, lane_id=0, initial_registers={0: 1, 1: 2, 2: 3}, max_steps=2)
                self.assertEqual(raised.exception.code, code)

    def test_plop3_remains_explicitly_outside_scalar_contract(self):
        sass = b"/*0000*/ PLOP3.LUT P0, PT, PT, PT, PT, 0x80, 0x0 ;\n/*0010*/ EXIT ;\n"
        with self.assertRaises(TraceRefusal) as raised:
            execute(sass, lane_id=0)
        self.assertEqual(raised.exception.code, "opcode_unsupported")

    def test_lop3_all_256_truth_tables_preserve_operand_order_and_pc(self):
        # These bit planes enumerate all eight truth-table rows four times.
        # Expected bytes are the LUT itself, not a second copy of the executor.
        for lut in range(256):
            with self.subTest(lut=lut):
                sass = f"/*0000*/ LOP3.LUT R4, R0, R2, R3, {lut}, !PT;\n/*0010*/ EXIT;\n".encode()
                result = execute(sass, lane_id=0, initial_registers={
                    0: 0xF0F0F0F0, 2: 0xCCCCCCCC, 3: 0xAAAAAAAA,
                }, max_steps=2)
                self.assertEqual(result["records"][0]["registers_after"]["R4"], lut * 0x01010101)
                self.assertEqual([record["pc"] for record in result["records"]], [0, 16])

    def test_lop3_matches_independent_boolean_expressions(self):
        rng = random.Random(86)
        inputs = [(0, 0xFFFFFFFF, 0x80000000), (0xFFFFFFFF, 0, 1)]
        inputs += [tuple(rng.getrandbits(32) for _ in range(3)) for _ in range(32)]
        expressions = {
            0xF0: lambda a, b, c: a,
            0xCC: lambda a, b, c: b,
            0xAA: lambda a, b, c: c,
            0x40: lambda a, b, c: a & b & ~c,
            0xCA: lambda a, b, c: (a & b) | (~a & c),
            0x1A: lambda a, b, c: ((a & b) | c) ^ a,
            0x96: lambda a, b, c: a ^ b ^ c,
        }
        for lut, expression in expressions.items():
            for a, b, c in inputs:
                with self.subTest(lut=lut, inputs=(a, b, c)):
                    sass = f"/*0000*/ LOP3.LUT R0, R0.reuse, R2, R3, {lut}, !PT;\n/*0010*/ EXIT;\n".encode()
                    result = execute(sass, lane_id=0, initial_registers={0: a, 2: b, 3: c}, max_steps=2)
                    self.assertEqual(result["records"][0]["registers_after"]["R0"], expression(a, b, c) & 0xFFFFFFFF)

    def test_lop3_rejects_negative_lut_and_unmodeled_native_forms(self):
        cases = [
            ("R4, R0, R2, R3, -1, !PT", "lut_unsupported"),
            ("R4, R0, R2, R3, 0x40", "operand_arity"),
            ("R4, R0, R2, R3, 0x40, P0", "lop3_predicate_unsupported"),
            ("R4, R0.HI, R2, R3, 0x40, !PT", "lop3_operand_unsupported"),
            ("R4, R0, R2, UR3, 0x40, !PT", "lop3_operand_unsupported"),
            ("R4, R0, 0x100000000, R3, 0x40, !PT", "lop3_operand_unsupported"),
            ("R4.reuse, R0, R2, R3, 0x40, !PT", "register_unsupported"),
            ("R4, R0, R256, R3, 0x40, !PT", "register_unsupported"),
        ]
        for operands, code in cases:
            with self.subTest(operands=operands):
                sass = f"/*0000*/ LOP3.LUT {operands};\n/*0010*/ EXIT;\n".encode()
                with self.assertRaises(TraceRefusal) as raised:
                    execute(sass, lane_id=0, initial_registers={0: 1, 2: 2, 3: 4}, max_steps=2)
                self.assertEqual(raised.exception.code, code)

    def test_lop3_missing_inputs_and_unknown_guard_refuse(self):
        for guard, code in [("", "register_value_missing"), ("@P0 ", "predicate_value_missing")]:
            sass = f"/*0000*/ {guard}LOP3.LUT R4, R0, R2, R3, 0x40, !PT;\n/*0010*/ EXIT;\n".encode()
            with self.assertRaises(TraceRefusal) as raised:
                execute(sass, lane_id=0, max_steps=2)
            self.assertEqual(raised.exception.code, code)

    def test_static_manifest_binds_sass_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({
                "schema": "sm86_static_sass_artifact_v1",
                "target": "sm_86",
                "static_sass": True,
                "artifacts": {"sass": {"sha256": hashlib.sha256(SASS).hexdigest()}},
            }))
            self.assertEqual(_load_manifest(manifest, SASS)["target"], "sm_86")
            with self.assertRaises(TraceRefusal) as raised:
                _load_manifest(manifest, SASS + b"\n")
            self.assertEqual(raised.exception.code, "sass_digest_mismatch")


if __name__ == "__main__":
    unittest.main()
