"""Small metric-access helper for ``read_ncu_report.py``.

The Nsight Compute Python report API has changed shape across CUDA
installs, and this script is a local-label utility.  Keep the helper
defensive: try common object/dict access paths and return ``0.0`` when a
metric is absent so the caller can still emit a spreadsheet.
"""
from __future__ import annotations

from typing import Any


def _coerce_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return 0.0
    for attr in ("value", "as_double", "as_float", "as_uint64", "as_int64"):
        member = getattr(value, attr, None)
        if member is None:
            continue
        try:
            raw = member() if callable(member) else member
        except TypeError:
            continue
        return _coerce_number(raw)
    return 0.0


def _lookup_metric(kernel: Any, metric_name: str) -> Any:
    if isinstance(kernel, dict):
        if metric_name in kernel:
            return kernel[metric_name]
        metrics = kernel.get("metrics")
        if isinstance(metrics, dict) and metric_name in metrics:
            return metrics[metric_name]

    for method in ("metric_by_name", "metric", "get_metric", "get"):
        getter = getattr(kernel, method, None)
        if getter is None:
            continue
        try:
            return getter(metric_name)
        except (KeyError, TypeError, AttributeError):
            continue

    try:
        return kernel[metric_name]
    except (KeyError, TypeError, AttributeError):
        return None


def _metric(kernel: Any, metric_name: str) -> float:
    return _coerce_number(_lookup_metric(kernel, metric_name))


def get_launch__occupancy_limit_blocks(kernel: Any) -> float:
    return _metric(kernel, "launch__occupancy_limit_blocks")


def get_launch__occupancy_limit_registers(kernel: Any) -> float:
    return _metric(kernel, "launch__occupancy_limit_registers")


def get_launch__occupancy_limit_shared_mem(kernel: Any) -> float:
    return _metric(kernel, "launch__occupancy_limit_shared_mem")


def get_launch__occupancy_limit_warps(kernel: Any) -> float:
    return _metric(kernel, "launch__occupancy_limit_warps")


def get_sm__maximum_warps_avg_per_active_cycle(kernel: Any) -> float:
    return _metric(kernel, "sm__maximum_warps_avg_per_active_cycle")


def get_sm__maximum_warps_per_active_cycle_pct(kernel: Any) -> float:
    return _metric(kernel, "sm__maximum_warps_per_active_cycle_pct")


def get_sm__warps_active_avg_per_cycle_active(kernel: Any) -> float:
    return _metric(kernel, "sm__warps_active.avg.per_cycle_active")


def get_sm__warps_active_avg_pct_of_peak_sustained_active(kernel: Any) -> float:
    return _metric(kernel, "sm__warps_active.avg.pct_of_peak_sustained_active")


def get_l1tex__t_sector_hit_rate_pct(kernel: Any) -> float:
    return _metric(kernel, "l1tex__t_sector_hit_rate.pct")


def get_l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate_pct(
    kernel: Any,
) -> float:
    return _metric(
        kernel,
        "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    )


def get_lts__t_sector_hit_rate_pct(kernel: Any) -> float:
    return _metric(kernel, "lts__t_sector_hit_rate.pct")


def get_l1tex__t_requests_pipe_lsu_mem_global_op_ld_sum(kernel: Any) -> float:
    return _metric(kernel, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")


def get_l1tex__t_requests_pipe_lsu_mem_global_op_st_sum(kernel: Any) -> float:
    return _metric(kernel, "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum")


def get_l1tex__t_sectors_pipe_lsu_mem_global_op_ld_sum(kernel: Any) -> float:
    return _metric(kernel, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")


def get_l1tex__t_sectors_pipe_lsu_mem_global_op_st_sum(kernel: Any) -> float:
    return _metric(kernel, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")


def get_l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld_sum(kernel: Any) -> float:
    return _metric(kernel, "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum")


def get_l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st_sum(
    kernel: Any,
) -> float:
    return _metric(kernel, "l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum")


def get_dram__sectors_read_sum(kernel: Any) -> float:
    return _metric(kernel, "dram__sectors_read.sum")


def get_dram__sectors_write_sum(kernel: Any) -> float:
    return _metric(kernel, "dram__sectors_write.sum")


def get_l1tex__t_requests_pipe_lsu_mem_global_op_atom_sum(kernel: Any) -> float:
    return _metric(kernel, "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum")


def get_l1tex__t_requests_pipe_lsu_mem_global_op_red_sum(kernel: Any) -> float:
    return _metric(kernel, "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum")


def get_gpc__cycles_elapsed_max(kernel: Any) -> float:
    return _metric(kernel, "gpc__cycles_elapsed.max")


def get_sm__cycles_active_avg(kernel: Any) -> float:
    return _metric(kernel, "sm__cycles_active.avg")


def get_smsp__inst_executed_sum(kernel: Any) -> float:
    return _metric(kernel, "smsp__inst_executed.sum")


def get_sm__inst_executed_avg_per_cycle_elapsed(kernel: Any) -> float:
    return _metric(kernel, "sm__inst_executed.avg.per_cycle_elapsed")


def get_gpc__cycles_elapsed_avg_per_second(kernel: Any) -> float:
    return _metric(kernel, "gpc__cycles_elapsed.avg.per_second")


def get_gpu__time_duration_sum(kernel: Any) -> float:
    return _metric(kernel, "gpu__time_duration.sum")


def get_lts__t_requests_srcunit_tex_sum(kernel: Any) -> float:
    return _metric(kernel, "lts__t_requests_srcunit_tex.sum")


def get_dram__bytes_sum_per_second(kernel: Any) -> float:
    return _metric(kernel, "dram__bytes.sum.per_second")


def get_dram__bytes_read_sum_per_second(kernel: Any) -> float:
    return _metric(kernel, "dram__bytes_read.sum.per_second")


def get_dram__throughput_avg_pct_of_peak_sustained_elapsed(kernel: Any) -> float:
    return _metric(kernel, "dram__throughput.avg.pct_of_peak_sustained_elapsed")


def get_gpu__compute_memory_access_throughput_avg_pct_of_peak_sustained_elapsed(
    kernel: Any,
) -> float:
    return _metric(
        kernel,
        "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    )


def get_gpu__compute_memory_request_throughput_avg_pct_of_peak_sustained_elapsed(
    kernel: Any,
) -> float:
    return _metric(
        kernel,
        "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed",
    )


def get_sm__memory_throughput_avg_pct_of_peak_sustained_elapsed(
    kernel: Any,
) -> float:
    return _metric(
        kernel,
        "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed",
    )


def get_dram__cycles_elapsed_avg_per_second(kernel: Any) -> float:
    return _metric(kernel, "dram__cycles_elapsed.avg.per_second")


def get_L1_Total_Requests(kernel: Any) -> float:
    # GA102 exposes the canonical unified counter directly. Older report
    # sections may omit it, so retain the complete 11-family fallback used by
    # the original HyFiSS helper rather than silently dropping local,
    # surface, or texture traffic.
    direct = _lookup_metric(kernel, "l1tex__t_requests.sum")
    if direct is not None:
        return _coerce_number(direct)
    return sum(
        _metric(kernel, name)
        for name in [
            "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_requests_pipe_tex_mem_surface_op_ld.sum",
            "l1tex__t_requests_pipe_tex_mem_texture.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
            "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum",
            "l1tex__t_requests_pipe_tex_mem_surface_op_st.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
            "l1tex__t_requests_pipe_tex_mem_surface_op_red.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
            "l1tex__t_requests_pipe_tex_mem_surface_op_atom.sum",
        ]
    )
