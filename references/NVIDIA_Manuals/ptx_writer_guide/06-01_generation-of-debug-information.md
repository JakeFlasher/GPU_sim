---
title: "6.1. Generation of Debug Information"
section: "6.1"
source: "https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#generation-of-debug-information"
---

## [6.1. Generation of Debug Information](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability#generation-of-debug-information)[](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/#generation-of-debug-information "Permalink to this headline")

The responsibility for generating debug information is split between the PTX producer and the PTX-to-SASS backend. The PTX producer is responsible for emitting binary DWARF into the PTX file, using the .section and .b8-.b16-.b32-and-.b64 directives in PTX. This should contain the .debug_info and .debug_abbrev sections, and possibly optional sections .debug_pubnames and .debug_aranges. These sections are standard DWARF2 sections that refer to labels and registers in the PTX.

The PTX-to-SASS backend is responsible for generating the .debug_line section from the .file and .loc directives in the PTX file. This section maps source lines to SASS addresses. The backend also generates the .debug_frame section.
