---
title: "3. Function Calling Sequence"
section: "3"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#function-calling-sequence"
---

# [3. Function Calling Sequence](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#function-calling-sequence)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#function-calling-sequence "Permalink to this headline")

This section describes the PTX-level function calling sequence, including register usage, stack-frame layout, and parameter passing. The PTX-level function calling sequence describes what gets represented in PTX to enable function calls. There is an abstraction at this level. Most of the details associated with the function calling sequence are handled at the SASS level.

PTX versions earlier than 2.0 do not conform to the ABI defined in this document, and cannot perform ABI compatible function calls. For the calling convention to work PTX version 2.0 or greater must be used.
