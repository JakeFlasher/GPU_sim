---
title: "7.2. Keeping Intermediate Phase Files"
section: "7.2"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#keeping-intermediate-phase-files"
---

## [7.2. Keeping Intermediate Phase Files](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#keeping-intermediate-phase-files)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#keeping-intermediate-phase-files "Permalink to this headline")

`nvcc` stores intermediate results by default into temporary files that are deleted immediately before it completes. The location of the temporary file directories used are, depending on the current platform, as follows:

**Windows**

  Value of environment variable `TEMP` is used. If it is not set, `C:\Windows\temp` is used instead.

**Other Platforms**

  Value of environment variable `TMPDIR` is used. If it is not set, `/tmp` is used instead.

Option `--keep` makes `nvcc` store these intermediate files in the current directory or in the directory specified by `--keep-dir` instead, with names as described in [Supported Phases](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#supported-phases).
