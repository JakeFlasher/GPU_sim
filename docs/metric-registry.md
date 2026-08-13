# Metric registry

`tools/local_labels/metric_registry.json` catalogs NCU metric names, units,
and parsing information used by capture and evaluation code.

The registry prevents unit and naming mistakes. It is not a permission system:
researchers may request additional NCU metrics directly and then add stable
ones to the catalog when reuse is valuable.

NCU measurements are comparison targets. They must not be copied into replay
features or architecture parameters for the same evaluated workload.
