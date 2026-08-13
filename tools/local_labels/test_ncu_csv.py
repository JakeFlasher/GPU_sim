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


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(NcuCsvError, match="missing NCU CSV columns"):
        parse_ncu_csv('"Metric Name","Metric Value"\n"a","1"\n')
