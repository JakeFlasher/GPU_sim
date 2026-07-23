---
title: "@kernel"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#kernel"
---

### [@kernel](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general#kernel)[](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/#kernel "Permalink to this headline")

Defines GPU kernel functions, compiled as specialized GPU symbols through dynamic compilation.

**Decorator Parameters**:

- `preprocessor`:
  - `True` (default) — Automatically expands Python loops/ifs into GPU-compatible IR operations.
  - `False` — Expects manual or simplified kernel implementations.

**Kernel Launch Parameters**:

- `grid`
Specifies the grid size as a list of integers.
- `block`
Specifies the block size as a list of integers.
- `cluster`
Specifies the cluster size as a list of integers.
- `smem`
Specifies the size of shared memory in bytes (integer).
  - `None` (default) — Automatically calculates the kernel’s shared memory usage via **utils.SmemAllocator**. Recommended unless manual control is required.
  - `int` — Manually specifies the size of shared memory in bytes.

**Additional Kernel Launch Parameters**:

- `fallback_cluster`
Specifies the minimum-guaranteed cluster size. When set, `cluster` becomes the **preferred** size, enabling graceful degradation when hardware cannot satisfy the preferred dimensions.
  - `None` (default) — No fallback; `cluster` is used directly.
  - `list[int]` — Three-element list [x, y, z].
- `max_number_threads`
Specifies the maximum thread count per block (**maxntid**).
  - `[0, 0, 0]` (default) — Auto-generate **reqntid** from `block`.
  - `list[int]` — Three-element list [x, y, z].
- `min_blocks_per_mp`
Specifies the minimum blocks per multiprocessor (**minctasm**).
  - `0` (default) — No minimum occupancy hint.
  - `int` — Minimum number of blocks per multiprocessor.
- `use_pdl`
Enables Programmatic Dependent Launch (PDL) to overlap dependent kernel launches in the same stream.
  - `False` (default) — PDL disabled.
  - `True` — PDL enabled.
- `cooperative`
Enables cooperative kernel launch; all thread blocks launch cooperatively with grid-wide synchronization support.
  - `False` (default) — Standard kernel launch.
  - `True` — Cooperative kernel launch.
