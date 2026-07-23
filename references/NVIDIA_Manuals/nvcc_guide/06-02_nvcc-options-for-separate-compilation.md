---
title: "6.2. NVCC Options for Separate Compilation"
section: "6.2"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-options-for-separate-compilation"
---

## [6.2. NVCC Options for Separate Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvcc-options-for-separate-compilation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-options-for-separate-compilation "Permalink to this headline")

CUDA works by embedding device code into host objects. In whole program compilation, it embeds executable device code into the host object. In separate compilation, we embed relocatable device code into the host object, and run `nvlink`, the device linker, to link all the device code together. The output of nvlink is then linked together with all the host objects by the host linker to form the final executable.

The generation of relocatable vs executable device code is controlled by the `--relocatable-device-code` option.

The `--compile` option is already used to control stopping a compile at a host object, so a new option `--device-c` is added that simply does `--relocatable-device-code=true --compile`.

To invoke just the device linker, the `--device-link` option can be used, which emits a host object containing the embedded executable device code. The output of that must then be passed to the host linker. Or:

```text
nvcc <objects>
```

can be used to implicitly call both the device and host linkers. This works because if the device linker does not see any relocatable code it does not do anything.

The following figure shows the flow.

![CUDA Separate Compilation Trajectory](images/nvcc-options-for-separate-compilation_1.png)

CUDA Separate Compilation Trajectory

There is also a `-r` option for doing a relocatable link that generates a new relocatable object containing relocatable device code.
