---
title: "4.1. Command Option Types and Notation"
section: "4.1"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#command-option-types-and-notation"
---

## [4.1. Command Option Types and Notation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#command-option-types-and-notation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#command-option-types-and-notation "Permalink to this headline")

Each `nvcc` option has a long name and a short name, which are interchangeable with each other. These two variants are distinguished by the number of hyphens that must precede the option name: long names must be preceded by two hyphens, while short names must be preceded by a single hyphen. For example, `-I` is the short name of `--include-path`. Long options are intended for use in build scripts, where the size of the option is less important than the descriptive value. In contrast, short options are intended for interactive use.

`nvcc` recognizes three types of command options: boolean options, single value options, and list options.

Boolean options do not have an argument; they are either specified on the command line or not. Single value options must be specified at most once, but list options may be repeated. Examples of each of these option types are, respectively: `--verbose` (switch to verbose mode), `--output-file` (specify output file), and `--include-path` (specify include path).

Single value options and list options must have arguments, which must follow the name of the option itself by either one of more spaces or an equals character. When a one-character short name such as `-I`, `-l`, and `-L` is used, the value of the option may also immediately follow the option itself without being seperated by spaces or an equal character. The individual values of list options may be separated by commas in a single instance of the option, or the option may be repeated, or any combination of these two cases.

Hence, for the two sample options mentioned above that may take values, the following notations are legal:

```text
-o file
```

```text
-o=file
```

```text
-Idir1,dir2 -I=dir3 -I dir4,dir5
```

Unless otherwise specified, long option names are used throughout this document. However, short names can be used instead of long names for the same effect.
