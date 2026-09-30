# SM86 pipeline-category characterization — 2026-09-25

This bounded probe isolates SASS categories before applying them to retained
workloads. It compiles eight tiny kernels for SHF, PRMT, predicate/select,
conversion, SHFL, MOV, BAR, and async-control forms, checks device results, and
profiles the launches with independent NCU aggregate logical-pipe counters.

The probe establishes a category distinction without fitting scored workloads:
`SHF` is funnel-shift SASS and is not `SHFL` (warp shuffle); BAR/branch-family
instructions also have a CBU/ADU domain distinct from ordinary LSU traffic.
The native logical projection now preserves this distinction and leaves
dynamic physical selection as an explicit confidence domain.

Inputs and command artifacts are retained under
`results/captures/sm86_pipeline_categories_20260925_v1/`. The known static
instruction populations and aggregate counter rows are retained as comparison
evidence only; no NCU values enter replay configuration. The counters do not
identify a dynamic per-instruction physical subunit or issue-cycle route.

NVIDIA documents that pipeline names can be logical or physical, that some
instructions dynamically choose among multiple pipelines, and that SHF and
SHFL are distinct Ampere SASS operations ([Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/),
[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)).
