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

    def test_label_projection_is_bound_to_admitted_bytes_and_reports_partial_access(
        self,
    ) -> None:
        output = json.loads(
            self.run_python(
                f"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import analyze_line_coordinates as a

a.REPETITIONS = (1, 2)
root = Path(tempfile.mkdtemp())
orientation = "first_line_early"

def payload(repetition):
    return {{
        "workload": a.CONTRACT.workload,
        "mode": "pair",
        "groups": a.CONTRACT.direct_groups,
        "order": a.base.expected_order(repetition),
        "orientation": orientation,
        "correct": True,
        "early_fanout": {{"issue_cycles": 10, "correct": True, "mismatches": 0}},
        "late_fanout": {{"issue_cycles": 10, "correct": True, "mismatches": 0}},
        "paired_issue_delta_cycles_per_group": 0.0,
    }}

def identity(path):
    content = path.read_bytes()
    return {{
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }}

provenance = {{}}
for repetition in a.REPETITIONS:
    path = a.direct_result_path(root, orientation, repetition)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload(repetition)), encoding="utf-8")
    provenance[(orientation, repetition)] = {{"result": identity(path)}}

changed = payload(2)
changed["early_fanout"]["issue_cycles"] = 11
a.direct_result_path(root, orientation, 2).write_text(
    json.dumps(changed), encoding="utf-8"
)
access = a.new_label_access_state()
try:
    a.read_issue_labels(root, orientation, provenance, access)
except a.base.AnalysisError as error:
    failure = str(error)
else:
    failure = "not rejected"
summary = a.summarize_label_access({{"identification_issue": access}})
print(json.dumps({{"error": failure, "access": access, "summary": summary}}, sort_keys=True))
"""
            )
        )
        self.assertIn("result bytes changed after provenance admission", output["error"])
        self.assertEqual(
            output["access"],
            {
                "state": "partial",
                "attempted_repetitions": 1,
                "completed_repetitions": 1,
                "expected_repetitions": 2,
            },
        )
        self.assertTrue(output["summary"]["identification_issue_read"])
        self.assertEqual(
            output["summary"]["failure_access_state"]["identification_issue"],
            output["access"],
        )

    def test_malformed_later_repeat_records_partial_label_access(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import analyze_line_coordinates as a

a.REPETITIONS = (1, 2)
root = Path(tempfile.mkdtemp())
orientation = "first_line_early"
provenance = {{}}
for repetition, content in (
    (1, json.dumps({{
        "workload": a.CONTRACT.workload,
        "mode": "pair",
        "groups": a.CONTRACT.direct_groups,
        "order": "ab",
        "orientation": orientation,
        "correct": True,
        "early_fanout": {{"issue_cycles": 10, "correct": True, "mismatches": 0}},
        "late_fanout": {{"issue_cycles": 10, "correct": True, "mismatches": 0}},
        "paired_issue_delta_cycles_per_group": 0.0,
    }})),
    (2, '{{"workload":"truncated"'),
):
    path = a.direct_result_path(root, orientation, repetition)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode("utf-8")
    path.write_bytes(payload)
    provenance[(orientation, repetition)] = {{"result": {{
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }}}}
access = a.new_label_access_state()
try:
    a.read_issue_labels(root, orientation, provenance, access)
except a.base.AnalysisError:
    pass
summary = a.summarize_label_access({{"identification_issue": access}})
print(json.dumps({{"access": access, "summary": summary}}, sort_keys=True))
"""
            )
        )
        self.assertEqual(output["access"]["state"], "partial")
        self.assertEqual(output["access"]["attempted_repetitions"], 2)
        self.assertEqual(output["access"]["completed_repetitions"], 1)
        self.assertTrue(output["summary"]["identification_issue_read"])

    def test_direct_counters_and_paired_deltas_reject_invalid_numbers(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import json
import sys
sys.path.insert(0, {str(HERE)!r})
import analyze_line_coordinates as a

bad_counters = [True, float("inf"), float("-inf"), float("nan"), 1.5, 3.0]
bad_deltas = [True, float("inf"), float("-inf"), float("nan")]
counter_rejections = 0
delta_rejections = 0
for value in bad_counters:
    try:
        a.direct_counter(value, "counter")
    except a.base.AnalysisError:
        counter_rejections += 1
for value in bad_deltas:
    try:
        a.paired_delta(value, "delta")
    except a.base.AnalysisError:
        delta_rejections += 1
print(json.dumps({{
    "counter_rejections": counter_rejections,
    "delta_rejections": delta_rejections,
    "integer_counter": a.direct_counter(3, "counter"),
    "finite_delta": a.paired_delta(-1.25, "delta"),
}}))
"""
            )
        )
        self.assertEqual(output["counter_rejections"], 6)
        self.assertEqual(output["delta_rejections"], 4)
        self.assertEqual(output["integer_counter"], 3)
        self.assertEqual(output["finite_delta"], -1.25)

    def test_label_readers_reject_boolean_counters_and_deltas(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import analyze_line_coordinates as a

a.REPETITIONS = (1,)
root = Path(tempfile.mkdtemp())
orientation = "first_line_early"

def base_payload():
    return {{
        "workload": a.CONTRACT.workload,
        "mode": "pair",
        "groups": a.CONTRACT.direct_groups,
        "order": "ab",
        "orientation": orientation,
        "correct": True,
        "early_fanout": {{
            "issue_cycles": 10, "wait_cycles": 20, "total_cycles": 30,
            "sum_consistent": True, "correct": True, "mismatches": 0,
        }},
        "late_fanout": {{
            "issue_cycles": 10, "wait_cycles": 20, "total_cycles": 30,
            "sum_consistent": True, "correct": True, "mismatches": 0,
        }},
        "paired_issue_delta_cycles_per_group": 0.0,
        "paired_wait_delta_cycles_per_group": 0.0,
    }}

def run(payload, wait):
    path = a.direct_result_path(root, orientation, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload).encode("utf-8")
    path.write_bytes(content)
    provenance = {{(orientation, 1): {{"result": {{
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }}}}}}
    access = a.new_label_access_state()
    try:
        (a.read_wait_labels if wait else a.read_issue_labels)(
            root, orientation, provenance, access
        )
    except a.base.AnalysisError:
        return True
    return False

cases = []
payload = base_payload()
payload["early_fanout"]["mismatches"] = False
cases.append(run(payload, False))
payload = base_payload()
payload["paired_issue_delta_cycles_per_group"] = False
cases.append(run(payload, False))
payload = base_payload()
payload["paired_wait_delta_cycles_per_group"] = False
cases.append(run(payload, True))
print(json.dumps(cases))
"""
            )
        )
        self.assertEqual(output, [True, True, True])

    def test_valid_label_access_summary_preserves_frozen_booleans(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import json
import sys
sys.path.insert(0, {str(HERE)!r})
import analyze_line_coordinates as a

states = {{
    "identification_issue": a.new_label_access_state(),
    "identification_wait": a.new_label_access_state(),
    "holdout_issue": a.new_label_access_state(),
    "holdout_wait": a.new_label_access_state(),
}}
print(json.dumps(a.summarize_label_access(states), sort_keys=True))
"""
            )
        )
        self.assertEqual(
            output,
            {
                "identification_issue_read": False,
                "identification_wait_read": False,
                "holdout_issue_read": False,
                "holdout_wait_read": False,
            },
        )

    def test_coordinate_smoke_authorization_is_exact(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import copy
import json
import sys
sys.path.insert(0, {str(HERE)!r})
import line_ownership_campaign_contract as c

binary = {{"path": "/tmp/probe", "sha256": "0" * 64, "size_bytes": 1}}
expected = c._coordinate_smoke_authorization(
    c.COORDINATE_CONTRACT,
    c.COORDINATE_CONTRACT.evidence_root,
    binary,
)
c.validate_smoke_authorization(
    expected,
    contract=c.COORDINATE_CONTRACT,
    root=c.COORDINATE_CONTRACT.evidence_root,
    binary=binary,
)
mutations = []
for field, value in (
    ("counts", {{"ncu_bundles": 1}}),
    ("smoke_gates", {{"ncu_contract_exact": True}}),
    ("label_reads", {{"identification_issue_read": False}}),
):
    candidate = copy.deepcopy(expected)
    candidate[field] = value
    mutations.append(candidate)
candidate = copy.deepcopy(expected)
candidate["extra"] = True
mutations.append(candidate)
for path, value in (
    (("schema_version",), True),
    (("counts", "ncu_bundles"), True),
    (("counts", "ncu_bundles"), 1.0),
    (("timing_promotion_authorized",), 0),
    (("binary", "size_bytes"), True),
    (("campaign_contract", "fixed_depth"), True),
):
    candidate = copy.deepcopy(expected)
    target = candidate
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    mutations.append(candidate)
rejected = 0
for candidate in mutations:
    try:
        c.validate_smoke_authorization(
            candidate,
            contract=c.COORDINATE_CONTRACT,
            root=c.COORDINATE_CONTRACT.evidence_root,
            binary=binary,
        )
    except c.ContractError:
        rejected += 1
print(json.dumps({{"accepted": True, "rejected": rejected}}))
"""
            )
        )
        self.assertEqual(output, {"accepted": True, "rejected": 10})

    def test_campaign_contract_rejects_boolean_and_float_counts(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import dataclasses
import json
import sys
sys.path.insert(0, {str(HERE)!r})
import line_ownership_campaign_contract as c

rejected = 0
for changes in ({{"fixed_depth": True}}, {{"trace_groups": 64.0}}):
    try:
        dataclasses.replace(c.COORDINATE_CONTRACT, **changes)
    except c.ContractError:
        rejected += 1
print(json.dumps({{"rejected": rejected}}))
"""
            )
        )
        self.assertEqual(output, {"rejected": 2})

    def test_all_coordinate_validators_use_type_strict_equality(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import json
import sys
sys.path.insert(0, {str(HERE)!r})
import analyze_line_coordinates as analyzer
import capture_line_ownership_execution as execution
import capture_line_ownership_ncu as ncu

expected = {{"count": 1, "flag": False, "nested": [1, 0]}}
mutated = {{"count": True, "flag": 0, "nested": [1.0, False]}}
print(json.dumps({{
    "contract": analyzer.CONTRACT.generation_id,
    "execution": execution.same_json(mutated, expected),
    "ncu": ncu.same_json(mutated, expected),
    "analysis": analyzer.base.same_json(mutated, expected),
    "static": execution.support.strict_json_equal(mutated, expected),
}}))
"""
            )
        )
        self.assertEqual(
            output,
            {
                "contract": "source_line_ownership_coordinates_v1",
                "execution": False,
                "ncu": False,
                "analysis": False,
                "static": False,
            },
        )

    def test_type_mutated_contract_binding_is_rejected(self) -> None:
        output = self.run_python(
            f"""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import line_ownership_campaign_contract as c

parent = Path(tempfile.mkdtemp())
root = parent / "root"
root.mkdir()
contract = dataclasses.replace(
    c.COORDINATE_CONTRACT,
    evidence_root=root,
    aggregate_report=parent / "aggregate.json",
)
c.configure_once(contract)
import capture_line_ownership_execution as execution
execution.configure_campaign(contract)
document = contract.to_json()
document["fixed_depth"] = True
(root / "campaign_contract.json").write_text(
    json.dumps(document), encoding="utf-8"
)
try:
    execution.validate_campaign_contract_binding(root)
except execution.CaptureError:
    print("rejected")
"""
        )
        self.assertEqual(output, "rejected")

    def test_both_full_runners_reject_truncated_authorization(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import line_ownership_campaign_contract as c

parent = Path(tempfile.mkdtemp())
root = parent / "root"
aggregate = parent / "aggregate.json"
contract = dataclasses.replace(
    c.COORDINATE_CONTRACT,
    evidence_root=root,
    aggregate_report=aggregate,
)
c.configure_once(contract)
import capture_line_ownership_execution as execution
import capture_line_ownership_ncu as ncu
execution.configure_campaign(contract)
ncu.configure_contract(contract)
root.mkdir()
(root / "campaign_contract.json").write_text(
    json.dumps(contract.to_json()), encoding="utf-8"
)
binary = {{"path": str(contract.binary), "sha256": "0" * 64, "size_bytes": 1}}
good = c._coordinate_smoke_authorization(contract, root, binary)

def write(value):
    aggregate.write_text(json.dumps(value), encoding="utf-8")

write(good)
accepted = 0
for validator in (
    lambda: execution.require_full_campaign_authorization(root, binary, aggregate),
    lambda: ncu.require_full_campaign_authorization(root, binary),
):
    validator()
    accepted += 1

bad = dict(good)
bad["smoke_gates"] = {{"ncu_contract_exact": True}}
write(bad)
rejected = 0
for validator in (
    lambda: execution.require_full_campaign_authorization(root, binary, aggregate),
    lambda: ncu.require_full_campaign_authorization(root, binary),
):
    try:
        validator()
    except (execution.CaptureError, ncu.CampaignError):
        rejected += 1
print(json.dumps({{"accepted": accepted, "rejected": rejected}}))
"""
            )
        )
        self.assertEqual(output, {"accepted": 2, "rejected": 2})

    def test_validate_only_does_not_create_missing_contract(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import line_ownership_campaign_contract as c

parent = Path(tempfile.mkdtemp())
root = parent / "missing-root"
contract = dataclasses.replace(
    c.COORDINATE_CONTRACT,
    evidence_root=root,
    aggregate_report=parent / "report.json",
)
c.configure_once(contract)
import capture_line_ownership_execution as execution
execution.configure_campaign(contract)
rc = execution.main(["--root", str(root), "--smoke", "--validate-only"])
print(json.dumps({{
    "return_code": rc,
    "root_exists": root.exists(),
    "contract_exists": (root / "campaign_contract.json").exists(),
}}))
"""
            )
        )
        self.assertEqual(
            output,
            {"return_code": 1, "root_exists": False, "contract_exists": False},
        )

    def test_evidence_tree_rejects_forbidden_symlink_without_opening_target(
        self,
    ) -> None:
        output = json.loads(
            self.run_python(
                f"""
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import line_ownership_campaign_contract as c

opened = []
def audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        opened.append(os.fsdecode(args[0]))
sys.addaudithook(audit)

placements = (
    "campaign_contract.json",
    "static/command.json",
    "direct/cell/repeat-1/result.json",
    "paired_ncu/cell/repeat-1/global_ld/command.json",
    "nvbit/cell/dynamic_trace.pb",
    "simulator/cell/replay-1.json",
    "analysis/smoke-authorization.json",
)
rejected = 0
targets_opened = 0
for index, relative in enumerate(placements):
    target = c.COORDINATE_CONTRACT.forbidden_paths[
        index % len(c.COORDINATE_CONTRACT.forbidden_paths)
    ]
    root = Path(tempfile.mkdtemp()) / f"root-{{index}}"
    link = root / relative
    link.parent.mkdir(parents=True)
    os.symlink(target, link)
    contract = dataclasses.replace(
        c.COORDINATE_CONTRACT,
        evidence_root=root,
        aggregate_report=root.parent / f"report-{{index}}.json",
    )
    before = len(opened)
    try:
        c.validate_evidence_tree(contract, root)
    except c.ContractError:
        rejected += 1
    target_text = str(target)
    targets_opened += sum(
        path == target_text or path.startswith(target_text + os.sep)
        for path in opened[before:]
    )
print(json.dumps({{
    "rejected": rejected,
    "targets": len(placements),
    "target_opens": targets_opened,
}}))
"""
            )
        )
        self.assertEqual(output["rejected"], output["targets"])
        self.assertEqual(output["target_opens"], 0)

    def test_safe_report_publication_never_follows_output_symlink(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import line_ownership_campaign_contract as c

root = Path(tempfile.mkdtemp())
target = root / "sentinel.json"
target.write_text("sentinel", encoding="utf-8")
link = root / "report.json"
os.symlink(target, link)
contract = dataclasses.replace(
    c.COORDINATE_CONTRACT,
    evidence_root=root,
    aggregate_report=link,
)
try:
    c.replace_artifact_text(contract, link, "changed")
except c.ContractError:
    rejected = True
else:
    rejected = False
print(json.dumps({{
    "rejected": rejected,
    "target": target.read_text(encoding="utf-8"),
}}))
"""
            )
        )
        self.assertEqual(output, {"rejected": True, "target": "sentinel"})

    def test_capture_validation_entrypoints_reject_child_symlink(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import line_ownership_campaign_contract as c

parent = Path(tempfile.mkdtemp())
root = parent / "root"
contract = dataclasses.replace(
    c.COORDINATE_CONTRACT,
    evidence_root=root,
    aggregate_report=parent / "aggregate.json",
)
c.configure_once(contract)
import capture_line_ownership_execution as execution
import capture_line_ownership_ncu as ncu
execution.configure_campaign(contract)
ncu.configure_contract(contract)
root.mkdir()
(root / "campaign_contract.json").write_text(
    json.dumps(contract.to_json()), encoding="utf-8"
)
os.symlink(c.V1_CONTRACT.evidence_root, root / "static")
execution_rc = execution.main([
    "--root", str(root), "--smoke", "--validate-only"
])
ncu_rc = ncu.main([
    "--root", str(root), "--smoke", "--validate-only"
])
print(json.dumps({{
    "execution": execution_rc,
    "ncu": ncu_rc,
    "link_preserved": (root / "static").is_symlink(),
}}))
"""
            )
        )
        self.assertEqual(
            output,
            {"execution": 1, "ncu": 1, "link_preserved": True},
        )

    def test_postcheck_symlink_swap_cannot_escape_safe_writer(self) -> None:
        output = json.loads(
            self.run_python(
                f"""
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, {str(HERE)!r})
import line_ownership_campaign_contract as c

parent = Path(tempfile.mkdtemp())
root = parent / "root"
target = parent / "target"
root.mkdir()
target.mkdir()
contract = dataclasses.replace(
    c.COORDINATE_CONTRACT,
    evidence_root=root,
    aggregate_report=parent / "aggregate.json",
)
c.configure_once(contract)
import capture_line_ownership_execution as execution
execution.configure_campaign(contract)
(root / "campaign_contract.json").write_text(
    json.dumps(contract.to_json()), encoding="utf-8"
)
c.validate_evidence_tree(contract, root)
os.symlink(target, root / "paired_ncu")
try:
    execution.support.atomic_write_json(
        root / "paired_ncu" / "escaped.json", {{"escaped": True}}
    )
except execution.CaptureError:
    rejected = True
else:
    rejected = False
print(json.dumps({{
    "rejected": rejected,
    "target_created": (target / "escaped.json").exists(),
}}))
"""
            )
        )
        self.assertEqual(
            output,
            {"rejected": True, "target_created": False},
        )

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
