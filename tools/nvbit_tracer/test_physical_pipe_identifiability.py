import unittest

from physical_pipe_identifiability import DynamicPipeRecord, IdentifiabilityRefusal, identify


def rec(ordinal: int, candidates: tuple[str, ...]) -> DynamicPipeRecord:
    return DynamicPipeRecord(7, ordinal * 16, 0, ordinal, candidates)


class PhysicalPipeIdentifiabilityTests(unittest.TestCase):
    def test_two_ambiguous_records_preserve_both_assignments(self):
        result = identify(
            [rec(0, ("ADU", "CBU")), rec(1, ("ADU", "CBU"))],
            {"ADU": 1, "CBU": 1},
        )
        self.assertEqual(result["status"], "aggregate_ambiguous")
        self.assertEqual(result["solution_count"], 2)
        self.assertEqual(result["entropy_bits"], 1.0)
        self.assertFalse(result["claims"]["native_route_recovered"])

    def test_singletons_can_be_aggregate_unique_without_silicon_claim(self):
        result = identify(
            [rec(0, ("ADU",)), rec(1, ("CBU",))],
            {"ADU": 1, "CBU": 1},
        )
        self.assertEqual(result["status"], "aggregate_unique")
        self.assertTrue(result["claims"]["aggregate_unique_only"])
        self.assertFalse(result["claims"]["native_route_recovered"])

    def test_no_aggregate_is_route_unobservable(self):
        result = identify([rec(0, ("ALU", "FMA"))], None)
        self.assertEqual(result["status"], "route_unobservable")
        self.assertIsNone(result["solution_count"])

    def test_inconsistent_total_refuses(self):
        with self.assertRaises(IdentifiabilityRefusal) as raised:
            identify([rec(0, ("ALU",))], {"ALU": 2})
        self.assertEqual(raised.exception.code, "aggregate_total_mismatch")

    def test_unknown_aggregate_pipe_refuses(self):
        with self.assertRaises(IdentifiabilityRefusal) as raised:
            identify([rec(0, ("ALU",))], {"FMA": 1})
        self.assertEqual(raised.exception.code, "aggregate_pipe_unknown")

    def test_candidate_empty_refuses(self):
        with self.assertRaises(IdentifiabilityRefusal) as raised:
            identify([rec(0, ())], {"ALU": 1})
        self.assertEqual(raised.exception.code, "candidate_domain_empty")


if __name__ == "__main__":
    unittest.main()
