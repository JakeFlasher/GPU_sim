from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


class LineCoordinateAnalyzerTest(unittest.TestCase):
    def run_python(self, source: str) -> str:
        return subprocess.run(
            [sys.executable, "-c", source],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        ).stdout.strip()

    def test_default_and_coordinate_contracts_are_distinct(self) -> None:
        prefix = f"import sys; sys.path.insert(0, {str(HERE)!r}); "
        default = json.loads(
            self.run_python(
                prefix
                + "import line_ownership_campaign_contract as c; "
                + "import json; print(json.dumps(c.current_contract().to_json(), sort_keys=True))"
            )
        )
        coordinate = json.loads(
            self.run_python(
                prefix
                + "import analyze_line_coordinates as a; import json; "
                + "print(json.dumps(a.CONTRACT.to_json(), sort_keys=True))"
            )
        )
        self.assertEqual(default["generation_id"], "source_line_ownership_v1")
        self.assertEqual(
            coordinate["generation_id"], "source_line_ownership_coordinates_v1"
        )
        self.assertNotEqual(default["evidence_root"], coordinate["evidence_root"])
        self.assertEqual(coordinate["ncu_shared_wavefronts_per_copy"], 20)
        self.assertEqual(coordinate["simulator_destination_wavefronts_per_copy"], 24)
        self.assertEqual(coordinate["bypass_wavefronts_per_copy"], 32)
        self.assertEqual(coordinate["destination_lines_per_copy"], 20)

    def test_coordinate_module_rejects_predecessor_paths(self) -> None:
        prefix = f"import sys; sys.path.insert(0, {str(HERE)!r}); "
        output = self.run_python(
            prefix
            + "import analyze_line_coordinates as a; "
            + "from line_ownership_campaign_contract import ContractError; "
            + "\ntry:\n a.CONTRACT.require_allowed_path(a.V1_CONTRACT.evidence_root)"
            + "\nexcept ContractError:\n print('rejected')"
        )
        self.assertEqual(output, "rejected")

    def test_identification_and_holdout_statuses_are_frozen(self) -> None:
        prefix = f"import sys; sys.path.insert(0, {str(HERE)!r}); "
        output = json.loads(
            self.run_python(
                prefix
                + "import analyze_line_coordinates as a, json; "
                + "print(json.dumps(["
                + "a.outcome(errors=[],missing=[],smoke_only=False,smoke_passed=True,identification={'signal':False},holdout=None)[0],"
                + "a.outcome(errors=[],missing=[],smoke_only=False,smoke_passed=True,identification={'signal':True},holdout={'signal':False})[0],"
                + "a.outcome(errors=[],missing=[],smoke_only=False,smoke_passed=True,identification={'signal':True},holdout={'signal':True})[0]"
                + "]))"
            )
        )
        self.assertEqual(
            output,
            [
                "complete_line_coordinates_identification_no_go",
                "complete_line_coordinates_holdout_no_go",
                "complete_line_coordinates_signal",
            ],
        )

    def test_issue_projection_does_not_decode_wait_fields(self) -> None:
        prefix = f"import sys; sys.path.insert(0, {str(HERE)!r}); "
        output = json.loads(
            self.run_python(
                prefix
                + "import analyze_line_coordinates as a, json, tempfile; "
                + "from pathlib import Path; "
                + "p=Path(tempfile.mkstemp()[1]); "
                + "p.write_text('{\"workload\":\"async_source_line_ownership\",'"
                + "'\"mode\":\"pair\",\"groups\":8192,\"order\":\"ab\",'"
                + "'\"orientation\":\"first_line_early\",\"correct\":true,'"
                + "'\"early_fanout\":{\"issue_cycles\":2,\"wait_cycles\":1e999999,'"
                + "'\"correct\":true,\"mismatches\":0},'"
                + "'\"late_fanout\":{\"issue_cycles\":2,\"wait_cycles\":1e999999,'"
                + "'\"correct\":true,\"mismatches\":0},'"
                + "'\"paired_issue_delta_cycles_per_group\":0.0}'); "
                + "v=a.read_direct_fields(p,a._ISSUE_FIELDS); "
                + "print(json.dumps(v,sort_keys=True))"
            )
        )
        self.assertEqual(output["early_fanout"]["issue_cycles"], 2)
        self.assertNotIn("wait_cycles", output["early_fanout"])
        self.assertEqual(output["paired_issue_delta_cycles_per_group"], 0.0)

    def test_aggregate_path_is_frozen_before_evidence_reads(self) -> None:
        prefix = f"import sys; sys.path.insert(0, {str(HERE)!r}); "
        output = self.run_python(
            prefix
            + "import analyze_line_coordinates as a; from pathlib import Path; "
            + "root=a.CONTRACT.evidence_root; "
            + "\ntry:\n a.run(root,root/'analysis/report.json',root/'analysis/report.md',Path('/tmp/not-the-coordinate-report.json'),smoke_only=True)"
            + "\nexcept a.base.AnalysisError:\n print('rejected')"
        )
        self.assertEqual(output, "rejected")

if __name__ == "__main__":
    unittest.main()
