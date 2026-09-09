import pytest

from tools.local_labels.ncu_csv import NcuCsvError, group_by_metric, parse_ncu_csv


def test_parse_raw_ncu_csv() -> None:
    values = parse_ncu_csv(
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","copy(float*)","gpc__cycles_elapsed.max","cycle","1,234"\n'
        '"0","copy(float*)","smsp__inst_executed.sum","inst","42"\n'
    )
    assert values[0].kernel_id == 0
    assert values[0].metric_value == 1234.0
    assert group_by_metric(values)["smsp__inst_executed.sum"] == [42.0]


def test_parse_unquoted_csv_fields() -> None:
    values = parse_ncu_csv(
        "ID,Kernel Name,Metric Name,Metric Unit,Metric Value\n"
        "0,copy(float*),smsp__inst_executed.sum,inst,42\n"
    )
    assert values[0].kernel_id == 0
    assert values[0].metric_value == 42.0


def test_parse_text_with_utf8_bom() -> None:
    values = parse_ncu_csv(
        "\ufeffID,Kernel Name,Metric Name,Metric Unit,Metric Value\n"
        "0,copy(float*),smsp__inst_executed.sum,inst,42\n"
    )
    assert values[0].kernel_id == 0


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(NcuCsvError, match="missing NCU CSV columns"):
        parse_ncu_csv('"Metric Name","Metric Value"\n"a","1"\n')


def test_parse_ncu_2026_wide_raw_page_csv() -> None:
    values = parse_ncu_csv(
        "==PROF== report preamble\n"
        '"ID","Process ID","Kernel Name","gpc__cycles_elapsed.max",'
        '"smsp__inst_executed.sum","device__attribute_display_name"\n'
        '"","","","cycle","inst",""\n'
        '"4","123","copy(float*)","1,234","42","RTX 3090 Ti"\n'
        '"5","123","copy(float*)","1,250","43","RTX 3090 Ti"\n'
    )
    assert [(value.kernel_id, value.metric_name, value.metric_value) for value in values] == [
        (4, "gpc__cycles_elapsed.max", 1234.0),
        (4, "smsp__inst_executed.sum", 42.0),
        (5, "gpc__cycles_elapsed.max", 1250.0),
        (5, "smsp__inst_executed.sum", 43.0),
    ]
    assert values[0].metric_unit == "cycle"


def test_wide_row_width_mismatch_is_rejected() -> None:
    with pytest.raises(NcuCsvError, match="row 3 width"):
        parse_ncu_csv(
            '"ID","Kernel Name","gpc__cycles_elapsed.max"\n'
            '"","","cycle"\n'
            '"0","kernel"\n'
        )
