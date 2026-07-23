---
title: "6.2. Command-line Options"
section: "6.2"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvprune-options"
---

## [6.2. Command-line Options](https://docs.nvidia.com/cuda/cuda-binary-utilities#nvprune-options)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvprune-options "Permalink to this headline")

[Table 10](https://docs.nvidia.com/cuda/cuda-binary-utilities/#nvprune-options-table) contains supported command-line options of `nvprune`, along with a description of what each option does. Each option has a long name and a short name, which can be used interchangeably.

| Option (long) | Option (short) | Description |
| --- | --- | --- |
| `--arch <gpu architecture name>,...` | `-arch` | Specify the name of the NVIDIA GPU architecture which will remain in the object or library. |
| `--generate-code` | `-gencode` | This option is same format as nvcc –generate-code option, and provides a way to specify multiple architectures which should remain in the object or library. Only the ‘code’ values are used as targets to match. Allowed keywords for this option: ‘arch’,’code’. |
| `--no-relocatable-elf` | `-no-relocatable-elf` | Don’t keep any relocatable ELF. |
| `--output-file` | `-o` | Specify name and location of the output file. |
| `--help` | `-h` | Print this help information on this tool. |
| `--options-file <file>,...` | `-optf` | Include command line options from specified file. |
| `--version` | `-V` | Print version information on this tool. |
