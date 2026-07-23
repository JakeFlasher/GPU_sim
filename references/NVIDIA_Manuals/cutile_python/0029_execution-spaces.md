---
title: "Execution Spaces"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#execution-spaces"
---

## [Execution Spaces](https://docs.nvidia.com/cuda/cutile-python#execution-spaces)[](https://docs.nvidia.com/cuda/cutile-python/#execution-spaces "Permalink to this headline")

cuTile code runs on one or more _targets_. A target is an execution environment
defined by its hardware resources and programming model.

The set of targets where a construct can be used is called its _execution space_.
cuTile defines three execution spaces:

- _Host code_ — all CPU targets.
- _SIMT code_ — all CUDA SIMT targets.
(Historically called _device code_; we avoid that term to prevent ambiguity.)
- _Tile code_ — all CUDA tile targets.

Some constructs span multiple execution spaces. For example,
[`cdiv()`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.cdiv.html#cuda.tile.cdiv "cuda.tile.cdiv") is usable in both host code and tile code.

A function whose decorator explicitly specifies its execution space is called
an _annotated function_.
