from tools.local_labels.metrics import MetricCatalogError, load_metrics, metric_names


def test_catalog_has_unique_reusable_metric_names() -> None:
    metrics = load_metrics()
    assert len({metric.key for metric in metrics}) == len(metrics)
    assert "gpc__cycles_elapsed.max" in metric_names(["exec_cycles_gpc_max"])


def test_unknown_key_is_clear() -> None:
    try:
        metric_names(["not-a-metric"])
    except MetricCatalogError as error:
        assert "unknown metric key" in str(error)
    else:
        raise AssertionError("unknown key accepted")
