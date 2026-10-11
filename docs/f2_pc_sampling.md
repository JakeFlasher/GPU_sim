# F2 — CUPTI PC/PM observation seam

**Primary collector:** CUDA 13.2 `pc_sampling_continuous` sample  
**Run record:** `/dev/shm/sm120_rtx_5090_work/f2_pc_sampling_result.json`

The installed CUPTI sample was built and invoked against the retained FP32
dependency-chain workload with serialized collection. It returned no sampling
data file on this SM120 cloud image (`unavailable_tool_or_sku`); the wrapper
retains stdout/stderr and falls back to aggregate NCU observations without
inventing PC rows, correlation IDs or launch-zero assignments. The existing
envelope parser is ready for either CUPTI CSV or utility JSON when a supported
driver/tool path is available.

This closes the collector integration seam and records the current tool
limitation. It does not claim that PC sampling is impossible on every SM120
driver, and it does not convert sampled stall populations into cycles.
