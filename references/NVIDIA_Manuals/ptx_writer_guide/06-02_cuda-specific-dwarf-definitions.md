---
title: "6.2. CUDA-Specific DWARF Definitions"
section: "6.2"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#cuda-specific-dwarf-definitions"
---

## [6.2. CUDA-Specific DWARF Definitions](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#cuda-specific-dwarf-definitions)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#cuda-specific-dwarf-definitions "Permalink to this headline")

In order to support debugging of multiple memory segments, address class codes are defined to reflect the memory space of variables. The address-class values are emitted as the DW_AT_address_class attribute for all variable and parameter Debugging Information Entries. The address class codes are defined in the below table.

| Code | Value | Description |
| --- | --- | --- |
| ADDR_code_space | 1 | Code storage |
| ADDR_reg_space | 2 | Register storage |
| ADDR_sreg_space | 3 | Special register storage |
| ADDR_const_space | 4 | Constant storage |
| ADDR_global_space | 5 | Global storage |
| ADDR_local_space | 6 | Local storage |
| ADDR_param_space | 7 | Parameter storage |
| ADDR_shared_space | 8 | Shared storage |
| ADDR_surf_space | 9 | Surface storage |
| ADDR_tex_space | 10 | Texture storage |
| ADDR_tex_sampler_space | 11 | Texture sampler storage |
| ADDR_generic_space | 12 | Generic-address storage |
