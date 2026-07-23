---
title: "4. Syntax"
section: "4"
source: "https://docs.nvidia.com/cuda/parallel-thread-execution/#syntax"
---

# [4. Syntax](https://docs.nvidia.com/cuda/parallel-thread-execution#syntax)[](https://docs.nvidia.com/cuda/parallel-thread-execution/#syntax "Permalink to this headline")

PTX programs are a collection of text source modules (files). PTX source modules have an
assembly-language style syntax with instruction operation codes and operands. Pseudo-operations
specify symbol and addressing management. The ptxas optimizing backend compiler optimizes and
assembles PTX source modules to produce corresponding binary object files.
