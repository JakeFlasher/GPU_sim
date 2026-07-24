"""Contracts for the independent NCU launch-identity observation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.local_labels import metric_registry
from tools.local_labels import ncu_action_executor as ncu_executor
from tools.local_labels import parse_ncu_metrics as parser


def _expected_ncu_identity() -> ncu_executor.ExpectedNcuIdentity:
    return ncu_executor.build_expected_ncu_identity(
        resolved_path="/opt/ncu",
        executable_sha256="a" * 64,
        version_line="Version 2026.1.0.0 (public-release)",
    )


def test_report_import_requests_mangled_kernel_names(monkeypatch, tmp_path):
    report = tmp_path / "launch.ncu-rep"
    report.write_bytes(b"synthetic report bytes")
    observed: dict[str, object] = {}
    csv_text = (
        '"ID","Kernel Name","Block Size","Grid Size","Device",'
        '"Metric Name","Metric Value"\n'
        '"0","_Z15syntheticKernelv","(128, 1, 1)",'
        '"(1, 1, 1)","0","gpc__cycles_elapsed.max","123"\n'
    )
    expected = _expected_ncu_identity()

    def fake_execute(identity, arguments, **kwargs):
        retained = kwargs["retained_report"]
        observed["identity"] = identity
        observed["arguments"] = list(arguments)
        observed["report_path"] = retained.path
        observed["report_proc_path"] = retained.proc_path
        observed["report_sha256"] = retained.sha256
        assert kwargs["timeout_seconds"] == parser._NCU_IMPORT_TIMEOUT_SECONDS
        return ncu_executor.NcuActionResult(0, csv_text.encode(), b"")

    monkeypatch.setattr(ncu_executor, "execute_ncu_action", fake_execute)
    imported, error = parser._import_report_csv(expected, report)

    assert error is None
    assert imported == parser.ImportedReportCsv(
        text=csv_text,
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        report_size_bytes=len(report.read_bytes()),
    )
    assert observed["identity"] == expected
    assert observed["report_path"] == report.absolute()
    assert observed["report_sha256"] == imported.report_sha256
    command = observed["arguments"]
    assert isinstance(command, list)
    assert command[0:2] == ["--import", observed["report_proc_path"]]
    assert command[1] != str(report)
    assert command.count("--print-kernel-base") == 1
    option = command.index("--print-kernel-base")
    assert command[option + 1] == "mangled"
    observation = parser.extract_launch_observation(imported.text)
    assert observation == {
        "identity_columns": [
            "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
        ],
        "identity": {
            "ID": "0",
            "Device": "0",
            "Kernel Name": "_Z15syntheticKernelv",
            "Block Size": "(128, 1, 1)",
            "Grid Size": "(1, 1, 1)",
        },
    }


def test_report_import_strips_only_closed_ncu_warning_preamble(
    monkeypatch, tmp_path,
):
    report = tmp_path / "launch.ncu-rep"
    report.write_bytes(b"synthetic report bytes")
    csv_text = (
        '"ID","Kernel Name","Block Size","Grid Size","Device",'
        '"Metric Name","Metric Value"\n'
        '"0","_Z15syntheticKernelv","(128, 1, 1)",'
        '"(1, 1, 1)","0","gpc__cycles_elapsed.max","123"\n'
    )
    output = (
        "==WARNING== Could not determine user home directory for section "
        "deployment.\n"
        "==WARNING== Falling back to stock installation sections.\n\n"
        + csv_text
    )
    monkeypatch.setattr(
        ncu_executor,
        "execute_ncu_action",
        lambda *_args, **_kwargs: ncu_executor.NcuActionResult(
            0, output.encode(), b""
        ),
    )
    imported, error = parser._import_report_csv(
        _expected_ncu_identity(), report
    )
    assert error is None
    assert imported is not None
    assert imported.text == csv_text


@pytest.mark.parametrize(
    "prefix",
    [
        "unexpected banner\n",
        "==ERROR== synthetic error\n",
        '"Metric Name","Metric Value"\n',
    ],
)
def test_report_import_rejects_unclosed_stdout_preamble(
    monkeypatch, tmp_path, prefix,
):
    report = tmp_path / "launch.ncu-rep"
    report.write_bytes(b"synthetic report bytes")
    csv_text = (
        '"ID","Kernel Name","Block Size","Grid Size","Device",'
        '"Metric Name","Metric Value"\n'
        '"0","kernel","1","1","0","metric","1"\n'
    )
    monkeypatch.setattr(
        ncu_executor,
        "execute_ncu_action",
        lambda *_args, **_kwargs: ncu_executor.NcuActionResult(
            0, (prefix + csv_text).encode(), b""
        ),
    )
    imported, error = parser._import_report_csv(
        _expected_ncu_identity(), report
    )
    assert imported is None
    assert error is not None
    assert error.typed_reason == "ncu_import_failed"


def test_parser_cli_bootstraps_repository_imports_without_pythonpath(tmp_path):
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "LC_ALL": "C",
    }
    result = subprocess.run(
        [sys.executable, str(Path(parser.__file__).resolve()), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--ncu-resolved-path" in result.stdout
    assert "--ncu-executable-sha256" in result.stdout
    assert "--ncu-version-line" in result.stdout


def test_report_import_rejects_malformed_authority_before_report_open(
    monkeypatch, tmp_path,
):
    report = tmp_path / "launch.ncu-rep"
    report.write_bytes(b"synthetic report bytes")

    def unexpected(*_args, **_kwargs):
        pytest.fail("malformed NCU authority reached report retention or execution")

    monkeypatch.setattr(parser.immutable, "open_retained_regular", unexpected)
    imported, error = parser._import_report_csv(
        ncu_executor.ExpectedNcuIdentity(
            resolved_path="ncu",
            executable_sha256="not-a-sha",
            version_line="not a Version line",
        ),
        report,
    )
    assert imported is None
    assert error is not None
    assert error.typed_reason == "ncu_import_authority_invalid"


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (
            ncu_executor.NCU_ACTION_INPUT_IDENTITY_INVALID,
            "ncu_import_report_identity_invalid",
        ),
        (
            ncu_executor.NCU_EXECUTABLE_IDENTITY_INVALID,
            "ncu_import_authority_invalid",
        ),
        (ncu_executor.NCU_VERSION_LINE_MISMATCH, "ncu_import_authority_invalid"),
        (ncu_executor.NCU_ACTION_TIMEOUT, "ncu_import_timeout"),
        (ncu_executor.NCU_ACTION_OUTPUT_LIMIT, "ncu_import_output_limit"),
        (ncu_executor.NCU_ACTION_LAUNCH_FAILED, "ncu_import_failed"),
    ],
)
def test_report_import_preserves_executor_failure_classification(
    monkeypatch, tmp_path, code, reason,
):
    report = tmp_path / "launch.ncu-rep"
    report.write_bytes(b"synthetic report bytes")

    def fail(*_args, **_kwargs):
        raise ncu_executor.NcuActionExecutorError(code, "synthetic failure")

    monkeypatch.setattr(ncu_executor, "execute_ncu_action", fail)
    imported, error = parser._import_report_csv(_expected_ncu_identity(), report)
    assert imported is None
    assert error is not None
    assert error.typed_reason == reason
    assert code in error.detail


def test_report_path_replacement_is_typed_as_input_identity_failure(
    monkeypatch, tmp_path,
):
    report = tmp_path / "launch.ncu-rep"
    report.write_bytes(b"original report")

    def replace_then_fail(_identity, _arguments, **_kwargs):
        retained = _kwargs["retained_report"]
        report.rename(tmp_path / "moved.ncu-rep")
        report.write_bytes(b"replacement report")
        with pytest.raises(parser.immutable.ImmutableArtifactError):
            retained.verify_unchanged()
        raise ncu_executor.NcuActionExecutorError(
            ncu_executor.NCU_ACTION_INPUT_IDENTITY_INVALID,
            "retained report changed",
        )

    monkeypatch.setattr(
        ncu_executor, "execute_ncu_action", replace_then_fail,
    )
    imported, error = parser._import_report_csv(_expected_ncu_identity(), report)
    assert imported is None
    assert error is not None
    assert error.typed_reason == "ncu_import_report_identity_invalid"


_REQUIRED_IDENTITY = {
    "ID": "0",
    "Kernel Name": "_Z15syntheticKernelv",
    "Block Size": "(128, 1, 1)",
    "Grid Size": "(1, 1, 1)",
    "Device": "0",
}


_REPORT_SHA256 = "a" * 64


def _sealed_observation() -> dict:
    columns = ["ID", "Device", "Kernel Name", "Block Size", "Grid Size"]
    document = {
        "schema_version": 1,
        "kind": "ncu_details_csv_launch_observation",
        "kernel_name_base": "mangled",
        "requested_launch_skip": 0,
        "requested_launch_count": 1,
        "report_sha256": _REPORT_SHA256,
        "identity_columns": columns,
        "identity": {column: _REQUIRED_IDENTITY[column] for column in columns},
    }
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    document["observation_sha256"] = hashlib.sha256(encoded).hexdigest()
    return document


def _reseal_observation(document: dict) -> None:
    unsigned = {
        key: value
        for key, value in document.items()
        if key != "observation_sha256"
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    document["observation_sha256"] = hashlib.sha256(encoded).hexdigest()


def test_validate_launch_observation_document_accepts_bound_producer_shape():
    parser.validate_launch_observation_document(
        _sealed_observation(),
        requested_launch_skip=0,
        requested_launch_count=1,
        report_sha256=_REPORT_SHA256,
    )


def test_validate_launch_observation_document_rejects_resealed_report_substitution():
    document = _sealed_observation()
    document["report_sha256"] = "b" * 64
    _reseal_observation(document)
    with pytest.raises(ValueError, match="authority binding"):
        parser.validate_launch_observation_document(
            document,
            requested_launch_skip=0,
            requested_launch_count=1,
            report_sha256=_REPORT_SHA256,
        )


def test_validate_launch_observation_document_rejects_reordered_identity_columns():
    document = _sealed_observation()
    document["identity_columns"].reverse()
    _reseal_observation(document)
    with pytest.raises(ValueError, match="identity columns differ"):
        parser.validate_launch_observation_document(
            document,
            requested_launch_skip=0,
            requested_launch_count=1,
            report_sha256=_REPORT_SHA256,
        )


def test_validate_launch_observation_document_rejects_invalid_seal():
    document = _sealed_observation()
    document["observation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="seal differs"):
        parser.validate_launch_observation_document(
            document,
            requested_launch_skip=0,
            requested_launch_count=1,
            report_sha256=_REPORT_SHA256,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", True),
        ("requested_launch_skip", False),
        ("requested_launch_count", True),
    ),
)
def test_validate_launch_observation_document_rejects_resealed_boolean_fields(
    field, value
):
    document = _sealed_observation()
    document[field] = value
    _reseal_observation(document)
    with pytest.raises(ValueError, match="authority binding"):
        parser.validate_launch_observation_document(
            document,
            requested_launch_skip=0,
            requested_launch_count=1,
            report_sha256=_REPORT_SHA256,
        )


def test_validate_launch_observation_document_rejects_boolean_selector():
    with pytest.raises(ValueError, match="selectors are invalid"):
        parser.validate_launch_observation_document(
            _sealed_observation(),
            requested_launch_skip=False,
            requested_launch_count=1,
            report_sha256=_REPORT_SHA256,
        )


def _observation_csv(*, omit: str | None = None, blank: str | None = None) -> str:
    fields = [name for name in _REQUIRED_IDENTITY if name != omit]
    values = {
        name: "" if name == blank else _REQUIRED_IDENTITY[name]
        for name in fields
    }
    return (
        ",".join(f'"{name}"' for name in fields)
        + "\n"
        + ",".join(f'"{values[name]}"' for name in fields)
        + "\n"
    )


@pytest.mark.parametrize("field", tuple(_REQUIRED_IDENTITY))
def test_launch_observation_rejects_missing_mandatory_column(field):
    with pytest.raises(ValueError, match="identity columns missing"):
        parser.extract_launch_observation(_observation_csv(omit=field))


@pytest.mark.parametrize("field", tuple(_REQUIRED_IDENTITY))
def test_launch_observation_rejects_duplicate_mandatory_column(field):
    fields = [*_REQUIRED_IDENTITY, field]
    values = [*(_REQUIRED_IDENTITY.values()), "forged"]
    text = (
        ",".join(f'"{name}"' for name in fields)
        + "\n"
        + ",".join(f'"{value}"' for value in values)
        + "\n"
    )
    with pytest.raises(ValueError, match="identity columns duplicate"):
        parser.extract_launch_observation(text)


@pytest.mark.parametrize("field", tuple(_REQUIRED_IDENTITY))
def test_launch_observation_rejects_blank_mandatory_value(field):
    with pytest.raises(ValueError, match="identity values blank"):
        parser.extract_launch_observation(_observation_csv(blank=field))


def test_launch_observation_rejects_multiple_ncu_identities():
    text = _observation_csv() + (
        '"1","_Z15syntheticKernelv","(128, 1, 1)",'
        '"(1, 1, 1)","0"\n'
    )
    with pytest.raises(ValueError, match="identity cardinality is 2"):
        parser.extract_launch_observation(text)


_LAUNCH_SET_FIELDS = [
    "Metric Value", "Stream", "ID", "Grid Size", "Metric Name",
    "Context", "Device", "Block Size", "Process ID", "Kernel Name",
]
_LAUNCH_IDENTITIES = {
    "7": {
        "ID": "7", "Process ID": "100", "Context": "1", "Stream": "2",
        "Device": "0", "Kernel Name": "_Z7kernelAv",
        "Block Size": "(128, 1, 1)", "Grid Size": "(4, 1, 1)",
    },
    "2": {
        "ID": "2", "Process ID": "100", "Context": "1", "Stream": "3",
        "Device": "0", "Kernel Name": "_Z7kernelBv",
        "Block Size": "(64, 1, 1)", "Grid Size": "(8, 1, 1)",
    },
    "11": {
        "ID": "11", "Process ID": "101", "Context": "2", "Stream": "1",
        "Device": "0", "Kernel Name": "_Z7kernelCv",
        "Block Size": "(32, 1, 1)", "Grid Size": "(2, 1, 1)",
    },
}


def _launch_row(
    report_id: str, metric: str, value: str,
    **identity_overrides: str,
) -> dict[str, str]:
    return {
        **_LAUNCH_IDENTITIES[report_id],
        **identity_overrides,
        "Metric Name": metric,
        "Metric Value": value,
    }


def _launch_set_csv(
    rows: list[dict[str, str]], *, fields: list[str] | None = None,
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields or _LAUNCH_SET_FIELDS,
        lineterminator="\n", extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _complete_metric_rows() -> list[dict[str, str]]:
    return [
        _launch_row("7", "lts__t_sector_hit_rate.pct", "101"),
        _launch_row("2", "lts__t_sector_hit_rate.pct", "88"),
        _launch_row("7", "l1tex__t_sector_hit_rate.pct", "70"),
        _launch_row("2", "l1tex__t_sector_hit_rate.pct", "70"),
        _launch_row("7", "gpc__cycles_elapsed.max", "123"),
        _launch_row("2", "gpc__cycles_elapsed.max", "456"),
        _launch_row("7", "l1tex__t_sector_hit_rate.pct", "70"),
    ]


def _launch_set_kwargs(expected: int = 2) -> dict[str, int]:
    return {
        "expected_launch_count": expected,
        "requested_launch_skip": 0,
        "requested_launch_count": expected + 1,
    }


def test_launch_set_observations_preserve_first_appearance_and_identity_order():
    rows = [
        _launch_row("7", "metric_a", "1"),
        _launch_row("2", "metric_a", "2"),
        _launch_row("7", "metric_b", "3"),
        _launch_row("11", "metric_a", "4"),
        _launch_row("2", "metric_b", "5"),
    ]
    text = _launch_set_csv(rows)
    observed = parser.extract_launch_observations(
        text, **_launch_set_kwargs(expected=3),
    )
    assert observed["identity_columns"] == [
        "ID", "Process ID", "Context", "Stream", "Device", "Kernel Name",
        "Block Size", "Grid Size",
    ]
    assert observed["launch_count"] == 3
    assert [
        launch["identity"]["ID"] for launch in observed["launches"]
    ] == ["7", "2", "11"]
    assert [
        launch["report_launch_ordinal"] for launch in observed["launches"]
    ] == [0, 1, 2]
    assert observed["launches"][0]["identity"] == _LAUNCH_IDENTITIES["7"]
    header, groups = parser._split_launch_set_csv(
        text, **_launch_set_kwargs(expected=3),
    )
    metric_index = header.index("Metric Name")
    assert [row[metric_index] for row in groups[0].rows] == [
        "metric_a", "metric_b",
    ]
    assert [row[metric_index] for row in groups[1].rows] == [
        "metric_a", "metric_b",
    ]


def test_launch_set_metric_split_isolates_launches_and_retains_diagnostics():
    registry = metric_registry.load_registry()
    observed = parser.extract_launch_metric_observations(
        _launch_set_csv(_complete_metric_rows()),
        registry,
        ["gpc__cycles_elapsed.max"],
        **_launch_set_kwargs(),
    )
    assert [row["report_id"] for row in observed] == ["7", "2"]
    assert observed[0]["fraction_values"] == {
        "lts__t_sector_hit_rate.pct": 1.0,
        "l1tex__t_sector_hit_rate.pct": 0.7,
    }
    assert observed[1]["fraction_values"] == {
        "lts__t_sector_hit_rate.pct": 0.88,
        "l1tex__t_sector_hit_rate.pct": 0.7,
    }
    assert observed[0]["unit_tagged_metrics"] == {
        "gpc__cycles_elapsed.max": {"unit": "cycles", "value": 123.0},
    }
    assert observed[1]["unit_tagged_metrics"] == {
        "gpc__cycles_elapsed.max": {"unit": "cycles", "value": 456.0},
    }
    assert observed[0]["parser_diagnostics"] == {
        "ncu_sector_hit_rate_clamped_to_100": [
            {"metric_label": "lts__t_sector_hit_rate.pct", "raw_value": 101.0},
        ],
    }
    assert observed[1]["parser_diagnostics"] == {}


def test_launch_set_metric_split_normalizes_identity_whitespace():
    rows = _complete_metric_rows()
    rows[0]["ID"] = " 7 "
    rows[2]["Kernel Name"] = "  _Z7kernelAv  "
    observed = parser.extract_launch_metric_observations(
        _launch_set_csv(rows), metric_registry.load_registry(),
        ["gpc__cycles_elapsed.max"], **_launch_set_kwargs(),
    )
    assert [row["report_id"] for row in observed] == ["7", "2"]


def test_launch_set_metric_split_permits_cross_family_metric_overlap():
    observed = parser.extract_launch_metric_observations(
        _launch_set_csv(_complete_metric_rows()),
        metric_registry.load_registry(),
        ["lts__t_sector_hit_rate.pct", "gpc__cycles_elapsed.max"],
        **_launch_set_kwargs(),
    )
    assert observed[0]["unit_tagged_metrics"][
        "lts__t_sector_hit_rate.pct"
    ] == {"unit": "pct", "value": 1.0}
    assert observed[0]["parser_diagnostics"] == {
        "ncu_sector_hit_rate_clamped_to_100": [
            {"metric_label": "lts__t_sector_hit_rate.pct", "raw_value": 101.0},
        ],
    }


def test_unsplit_launch_set_still_fails_single_launch_metric_guards():
    text = _launch_set_csv(_complete_metric_rows())
    percentage = parser.parse_raw_page_csv(
        text, ["l1tex__t_sector_hit_rate.pct"],
    )
    unit = parser.extract_unit_tagged_metrics(
        text, metric_registry.load_registry(), ["gpc__cycles_elapsed.max"],
    )
    assert not percentage.ok
    assert percentage.typed_reason == "ncu_csv_duplicate_metric_rows"
    assert not unit.ok
    assert unit.typed_reason == "ncu_csv_duplicate_metric_rows"


@pytest.mark.parametrize("field", parser._LAUNCH_SET_IDENTITY_COLUMNS)
def test_launch_set_observations_reject_missing_identity_header(field):
    fields = [name for name in _LAUNCH_SET_FIELDS if name != field]
    with pytest.raises(ValueError, match="identity columns missing"):
        parser.extract_launch_observations(
            _launch_set_csv(
                [_launch_row("7", "metric", "1")], fields=fields,
            ),
            **_launch_set_kwargs(expected=1),
        )


@pytest.mark.parametrize("field", parser._LAUNCH_SET_IDENTITY_COLUMNS)
def test_launch_set_observations_reject_duplicate_identity_header(field):
    with pytest.raises(ValueError, match="identity columns duplicate"):
        parser.extract_launch_observations(
            _launch_set_csv(
                [_launch_row("7", "metric", "1")],
                fields=[*_LAUNCH_SET_FIELDS, field],
            ),
            **_launch_set_kwargs(expected=1),
        )


@pytest.mark.parametrize("field", ["Metric Name", "Metric Value"])
def test_launch_set_observations_reject_missing_metric_header(field):
    fields = [name for name in _LAUNCH_SET_FIELDS if name != field]
    with pytest.raises(ValueError, match="metric columns missing"):
        parser.extract_launch_observations(
            _launch_set_csv(
                [_launch_row("7", "metric", "1")], fields=fields,
            ),
            **_launch_set_kwargs(expected=1),
        )


@pytest.mark.parametrize("field", ["Metric Name", "Metric Value"])
def test_launch_set_observations_reject_duplicate_metric_header(field):
    with pytest.raises(ValueError, match="metric columns duplicate"):
        parser.extract_launch_observations(
            _launch_set_csv(
                [_launch_row("7", "metric", "1")],
                fields=[*_LAUNCH_SET_FIELDS, field],
            ),
            **_launch_set_kwargs(expected=1),
        )


@pytest.mark.parametrize("field", parser._LAUNCH_SET_IDENTITY_COLUMNS)
def test_launch_set_observations_reject_blank_identity(field):
    with pytest.raises(ValueError, match="identity values blank"):
        parser.extract_launch_observations(
            _launch_set_csv([
                _launch_row("7", "metric", "1", **{field: "  "}),
            ]),
            **_launch_set_kwargs(expected=1),
        )


@pytest.mark.parametrize(
    "field",
    [
        name for name in parser._LAUNCH_SET_IDENTITY_COLUMNS if name != "ID"
    ],
)
def test_launch_set_observations_reject_repeated_id_identity_drift(field):
    with pytest.raises(ValueError, match=rf"identity drift.*{field}"):
        parser.extract_launch_observations(
            _launch_set_csv([
                _launch_row("7", "metric_a", "1"),
                _launch_row("7", "metric_b", "2", **{field: "forged"}),
            ]),
            **_launch_set_kwargs(expected=1),
        )


def test_launch_set_observations_reject_malformed_row_width():
    text = _launch_set_csv([_launch_row("7", "metric", "1")])
    header, row = text.splitlines()
    malformed = header + "\n" + row.rsplit(",", 1)[0] + "\n"
    with pytest.raises(ValueError, match="has 9 columns; expected 10"):
        parser.extract_launch_observations(
            malformed, **_launch_set_kwargs(expected=1),
        )


def test_launch_set_observations_reject_malformed_csv_quoting():
    text = _launch_set_csv([_launch_row("7", "metric", "1")])
    malformed = text.replace("_Z7kernelAv", '"_Z7kernelAv')
    with pytest.raises(ValueError, match="CSV is malformed"):
        parser.extract_launch_observations(
            malformed, **_launch_set_kwargs(expected=1),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_launch_count": True, "requested_launch_skip": 0,
         "requested_launch_count": 2},
        {"expected_launch_count": 0, "requested_launch_skip": 0,
         "requested_launch_count": 1},
        {"expected_launch_count": -1, "requested_launch_skip": 0,
         "requested_launch_count": 0},
        {"expected_launch_count": "1", "requested_launch_skip": 0,
         "requested_launch_count": 2},
        {"expected_launch_count": 1, "requested_launch_skip": "0",
         "requested_launch_count": 2},
        {"expected_launch_count": 1, "requested_launch_skip": 0,
         "requested_launch_count": "2"},
        {"expected_launch_count": 1, "requested_launch_skip": True,
         "requested_launch_count": 2},
        {"expected_launch_count": 1, "requested_launch_skip": 0,
         "requested_launch_count": True},
    ],
)
def test_launch_set_observations_reject_noncanonical_selector_types(kwargs):
    with pytest.raises(ValueError, match="selectors require strict integers"):
        parser.extract_launch_observations(
            _launch_set_csv([_launch_row("7", "metric", "1")]), **kwargs,
        )


@pytest.mark.parametrize(
    ("expected", "requested", "message"),
    [
        (1, 3, "count must equal expected count plus one"),
        (1, 1, "count must equal expected count plus one"),
        (1, -1, "count must equal expected count plus one"),
        (1, 2, "cardinality is 2; expected 1"),
        (3, 4, "cardinality is 2; expected 3"),
    ],
)
def test_launch_set_observations_reject_selector_or_observed_count_drift(
    expected, requested, message,
):
    with pytest.raises(ValueError, match=message):
        parser.extract_launch_observations(
            _launch_set_csv([
                _launch_row("7", "metric", "1"),
                _launch_row("2", "metric", "2"),
            ]),
            expected_launch_count=expected,
            requested_launch_skip=0,
            requested_launch_count=requested,
        )


@pytest.mark.parametrize("launch_skip", [1, -1])
def test_launch_set_observations_reject_nonzero_launch_skip(launch_skip):
    with pytest.raises(ValueError, match="launch skip must be 0"):
        parser.extract_launch_observations(
            _launch_set_csv([_launch_row("7", "metric", "1")]),
            expected_launch_count=1,
            requested_launch_skip=launch_skip,
            requested_launch_count=2,
        )


@pytest.mark.parametrize(
    ("percentage", "unit", "family"),
    [
        (["lts__t_sector_hit_rate.pct"] * 2,
         ["gpc__cycles_elapsed.max"], "percentage"),
        (["lts__t_sector_hit_rate.pct"],
         ["gpc__cycles_elapsed.max"] * 2, "unit-tagged"),
    ],
)
def test_launch_set_metric_split_rejects_duplicate_requests(
    percentage, unit, family,
):
    with pytest.raises(ValueError, match=rf"{family}.*duplicates"):
        parser.extract_launch_metric_observations(
            _launch_set_csv(_complete_metric_rows()),
            metric_registry.load_registry(), unit,
            percentage_metric_names=percentage,
            **_launch_set_kwargs(),
        )


@pytest.mark.parametrize(
    ("mutation", "unit_metrics", "reason", "ordinal"),
    [
        ("missing_unit", ["gpc__cycles_elapsed.max"],
         "ncu_csv_metric_missing", 1),
        ("conflicting_pct", ["gpc__cycles_elapsed.max"],
         "ncu_csv_duplicate_metric_rows", 1),
        ("nonnumeric_pct", ["gpc__cycles_elapsed.max"],
         "ncu_csv_metric_non_numeric", 1),
        ("out_of_range_pct", ["gpc__cycles_elapsed.max"],
         "ncu_csv_metric_out_of_range", 1),
        ("unregistered_unit", ["synthetic__unknown.sum"],
         "ncu_csv_metric_unregistered", 0),
    ],
)
def test_launch_set_metric_split_retains_child_typed_failure(
    mutation, unit_metrics, reason, ordinal,
):
    rows = _complete_metric_rows()
    if mutation == "missing_unit":
        rows = [
            row for row in rows
            if not (
                row["ID"] == "2"
                and row["Metric Name"] == "gpc__cycles_elapsed.max"
            )
        ]
    elif mutation == "conflicting_pct":
        rows.append(_launch_row("2", "lts__t_sector_hit_rate.pct", "77"))
    elif mutation in {"nonnumeric_pct", "out_of_range_pct"}:
        replacement = "not-a-number" if mutation == "nonnumeric_pct" else "106"
        for row in rows:
            if row["ID"] == "2" and row["Metric Name"] == (
                "lts__t_sector_hit_rate.pct"
            ):
                row["Metric Value"] = replacement
                break
    with pytest.raises(
        ValueError, match=rf"ordinal {ordinal}.*{reason}",
    ):
        parser.extract_launch_metric_observations(
            _launch_set_csv(rows), metric_registry.load_registry(), unit_metrics,
            **_launch_set_kwargs(),
        )


def test_launch_set_parser_is_process_and_filesystem_pure(monkeypatch, tmp_path):
    registry = metric_registry.load_registry()

    def unexpected(*_args, **_kwargs):
        pytest.fail("pure launch-set parsing reached process or filesystem I/O")

    monkeypatch.setattr(ncu_executor, "execute_ncu_action", unexpected)
    monkeypatch.setattr(parser.Path, "open", unexpected)
    monkeypatch.setattr(parser.Path, "write_text", unexpected)
    monkeypatch.setattr(parser.Path, "write_bytes", unexpected)
    monkeypatch.setattr("builtins.open", unexpected)
    parser.extract_launch_observations(
        _launch_set_csv(_complete_metric_rows()), **_launch_set_kwargs(),
    )
    parser.extract_launch_metric_observations(
        _launch_set_csv(_complete_metric_rows()),
        registry, ["gpc__cycles_elapsed.max"],
        **_launch_set_kwargs(),
    )
    assert list(tmp_path.iterdir()) == []
