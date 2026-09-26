from tools.local_labels.ncu_report_helper import (
    get_L1_Total_Requests,
    get_dram__bytes_sum_per_second,
    get_dram__bytes_read_sum_per_second,
    get_dram__cycles_elapsed_avg_per_second,
    get_dram__throughput_avg_pct_of_peak_sustained_elapsed,
    get_gpu__compute_memory_access_throughput_avg_pct_of_peak_sustained_elapsed,
    get_gpu__compute_memory_request_throughput_avg_pct_of_peak_sustained_elapsed,
    get_sm__memory_throughput_avg_pct_of_peak_sustained_elapsed,
)


def test_l1_total_requests_prefers_canonical_ga102_counter() -> None:
    kernel = {
        "metrics": {
            "l1tex__t_requests.sum": 81,
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": 10,
            "lts__t_requests_srcunit_tex.sum": 77,
        }
    }
    assert get_L1_Total_Requests(kernel) == 81.0


def test_l1_total_requests_falls_back_to_all_request_families() -> None:
    names = [
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
    kernel = {"metrics": {name: 1 for name in names}}
    assert get_L1_Total_Requests(kernel) == 11.0


def test_hyfiss_profiler_throughput_accessors_are_preserved() -> None:
    kernel = {
        "metrics": {
            "dram__bytes.sum.per_second": 1.25e9,
            "dram__bytes_read.sum.per_second": 2.5e8,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 37.5,
            "dram__cycles_elapsed.avg.per_second": 1.5e9,
            "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed": 10,
            "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed": 20,
            "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed": 30,
        }
    }
    assert get_dram__bytes_sum_per_second(kernel) == 1.25e9
    assert get_dram__bytes_read_sum_per_second(kernel) == 2.5e8
    assert get_dram__throughput_avg_pct_of_peak_sustained_elapsed(kernel) == 37.5
    assert get_dram__cycles_elapsed_avg_per_second(kernel) == 1.5e9
    assert (
        get_gpu__compute_memory_access_throughput_avg_pct_of_peak_sustained_elapsed(kernel)
        == 10.0
    )
    assert (
        get_gpu__compute_memory_request_throughput_avg_pct_of_peak_sustained_elapsed(kernel)
        == 20.0
    )
    assert get_sm__memory_throughput_avg_pct_of_peak_sustained_elapsed(kernel) == 30.0
