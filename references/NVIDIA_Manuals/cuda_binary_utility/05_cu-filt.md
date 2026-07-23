---
title: "5. cu++filt"
section: "5"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#cu-filt"
---

# [5. cu++filt](https://docs.nvidia.com/cuda/cuda-binary-utilities#cu-filt)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cu-filt "Permalink to this headline")

*cu++filt* decodes (demangles) low-level identifiers that have been mangled by CUDA C++ into user readable names. For every input alphanumeric word, the output of `cu++filt` is either the demangled name if the name decodes to a CUDA C++ name, or the original name itself.
