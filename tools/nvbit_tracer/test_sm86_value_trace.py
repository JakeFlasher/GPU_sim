#!/usr/bin/env python3
import unittest
import hashlib
import json
import tempfile
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
