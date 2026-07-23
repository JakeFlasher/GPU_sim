---
title: "Control Flow"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#control-flow"
---

### [Control Flow](https://docs.nvidia.com/cuda/cutile-python#control-flow)[](https://docs.nvidia.com/cuda/cutile-python/#control-flow "Permalink to this headline")

Python control flow statements (`if`, `for`, `while`, etc.) are usable in [tile code](https://docs.nvidia.com/cuda/cutile-python/#tile-code)
and can be arbitrarily nested.

#### [Current limitations](https://docs.nvidia.com/cuda/cutile-python#current-limitations)[](https://docs.nvidia.com/cuda/cutile-python/#current-limitations "Permalink to this headline")

[Tile code](https://docs.nvidia.com/cuda/cutile-python/#tile-code) imposes additional restrictions on control flow:

- `step` must be strictly positive.

Negative-step ranges such as `range(10, 0, -1)` are not supported today.
Passing a negative step indirectly via a variable may cause undefined
behavior.
