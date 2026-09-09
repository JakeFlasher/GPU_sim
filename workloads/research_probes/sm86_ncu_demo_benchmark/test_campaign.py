import csv
import io
from pathlib import Path

import pytest

from workloads.research_probes.sm86_ncu_demo_benchmark.campaign import _parse_ncu_metrics


@pytest.mark.parametrize("quoting", [csv.QUOTE_ALL, csv.QUOTE_MINIMAL])
def test_ncu_metric_identity_join_accepts_standard_csv(
    tmp_path: Path, quoting: int,
) -> None:
    stream = io.StringIO()
    stream.write("==PROF== report preamble\n")
    csv.writer(stream, quoting=quoting).writerows(
        [
            ["ID", "Process Name", "Kernel Name", "Grid Size", "Block Size", "Device", "CC", "smsp__inst_executed.sum"],
            ["", "", "", "", "", "", "", "inst"],
            ["4", "copy_run", "copy(float*)", "(2, 1, 1)", "(32, 1, 1)", "0", "8.6", "42"],
        ]
    )
    path = tmp_path / "ncu.csv"
    path.write_text(stream.getvalue())

    metrics, identity = _parse_ncu_metrics(path, ["smsp__inst_executed.sum"])

    assert metrics["smsp__inst_executed.sum"]["value"] == 42
    assert identity == {
        "ID": "4",
        "Process Name": "copy_run",
        "Kernel Name": "copy(float*)",
        "Grid Size": "(2, 1, 1)",
        "Block Size": "(32, 1, 1)",
        "Device": "0",
        "CC": "8.6",
    }
