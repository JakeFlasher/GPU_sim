---
title: "5.2. Command-line Options"
section: "5.2"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuplusplusfilt-options"
---

## [5.2. Command-line Options](https://docs.nvidia.com/cuda/cuda-binary-utilities#cuplusplusfilt-options)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuplusplusfilt-options "Permalink to this headline")

[Table 9](https://docs.nvidia.com/cuda/cuda-binary-utilities/#cuplusplusfilt-options-table) contains supported command-line options of `cu++filt`, along with a description of what each option does.

| Option | Description |
| --- | --- |
| `-_` | Strip underscore. On some systems, the CUDA compiler puts an underscore in front of every name. This option removes the initial underscore. Whether cu++filt removes the underscore by default is target dependent. |
| `-p` | When demangling the name of a function, do not display the types of the function’s parameters. |
| `-h` | Print a summary of the options to cu++filt and exit. |
| `-v` | Print the version information of this tool. |
