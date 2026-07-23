---
title: "4.2. Comments"
section: "4.2"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#comments"
---

## [4.2. Comments](https://docs.nvidia.com/cuda/parallel-thread-execution#comments)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#comments "Permalink to this headline")

Comments in PTX follow C/C++ syntax, using non-nested `/*` and `*/` for comments that may span
multiple lines, and using `//` to begin a comment that extends up to the next newline character,
which terminates the current line. Comments cannot occur within character constants, string
literals, or within other comments.

Comments in PTX are treated as whitespace.
