#!/usr/bin/env python3
import unittest

from register_value_adapter import Refusal, parse


class RegisterValueAdapterTests(unittest.TestCase):
    def test_parses_bounded_lane_snapshot_and_preserves_partial_identity(self):
        result = parse(
            """CTA 0,0,0 - warp 3 - IADD3 R0, R1, 1:
* Reg0_T0: 0x00000001 Reg0_T1: 0xffffffff
* Reg1_T0: 0x00000002 Reg1_T1: 0x00000003
"""
        )
        self.assertEqual(result["schema"], "nvbit_register_value_capture_v1")
        self.assertEqual(result["records"][0]["cta"], [0, 0, 0])
        self.assertEqual(result["records"][0]["operands"][0]["lanes"][1]["value_u32"], 0xFFFFFFFF)
        self.assertEqual(result["identity"]["status"], "partial")
        self.assertFalse(result["claims"]["native_sass_regeneration"])

    def test_refuses_duplicate_lane_values(self):
        with self.assertRaises(Refusal) as context:
            parse(
                """CTA 0,0,0 - warp 0 - MOV R0, R1:
* Reg0_T0: 0x00000001 Reg0_T0: 0x00000002
"""
            )
        self.assertEqual(context.exception.code, "duplicate_lane_value")

    def test_static_index_join_is_unique_or_explicitly_ambiguous(self):
        capture = """CTA 0,0,0 - warp 0 - MOV R0, R1:
* Reg0_T0: 0x00000001
"""
        unique = parse(
            capture,
            {"MOV R0, R1": [{"function_unique_id": 7, "pc": 32}]},
        )
        self.assertEqual(unique["records"][0]["static_identity"]["pc"], 32)
        self.assertEqual(unique["identity"]["static_join"]["unique"], 1)
        ambiguous = parse(
            capture,
            {"MOV R0, R1": [{"pc": 32}, {"pc": 48}]},
        )
        self.assertEqual(
            ambiguous["records"][0]["static_identity"]["status"], "ambiguous"
        )
        self.assertEqual(ambiguous["identity"]["static_join"]["ambiguous"], 1)

    def test_exact_dynamic_index_upgrades_to_value_trace_identity(self):
        capture = """CTA 0,0,0 - warp 0 - IADD3 R0, R1, 1:
* Reg0_T0: 0x00000001
"""
        result = parse(
            capture,
            dynamic_index={
                0: {
                    "sequence": 0,
                    "function_unique_id": 7,
                    "pc": 32,
                    "opcode": "IADD3 R0, R1, 1",
                }
            },
        )
        self.assertEqual(result["schema"], "nvbit_value_trace_capture_v1")
        self.assertEqual(result["identity"]["status"], "dynamic_joined")
        self.assertTrue(result["claims"]["value_dependent_dynamic_trace"])
        self.assertEqual(result["records"][0]["dynamic_identity"]["pc"], 32)

    def test_dynamic_index_refuses_opcode_mismatch(self):
        with self.assertRaises(Refusal) as context:
            parse(
                """CTA 0,0,0 - warp 0 - IADD3 R0, R1, 1:
* Reg0_T0: 0x00000001
""",
                dynamic_index={
                    0: {
                        "sequence": 0,
                        "function_unique_id": 7,
                        "pc": 32,
                        "opcode": "FADD R0, R1, R2",
                    }
                },
            )
        self.assertEqual(context.exception.code, "dynamic_identity_opcode_mismatch")


if __name__ == "__main__":
    unittest.main()
