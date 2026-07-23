---
title: "9.2. Bandwidth"
section: "9.2"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#bandwidth"
---

## [9.2. Bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#bandwidth)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#bandwidth "Permalink to this headline")

Bandwidth - the rate at which data can be transferred - is one of the most important gating factors for performance. Almost all changes to code should be made in the context of how they affect bandwidth. As described in [Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-optimizations) of this guide, bandwidth can be dramatically affected by the choice of memory in which data is stored, how the data is laid out and the order in which it is accessed, as well as other factors.

To measure performance accurately, it is useful to calculate theoretical and effective bandwidth. When the latter is much lower than the former, design or implementation details are likely to reduce bandwidth, and it should be the primary goal of subsequent optimization efforts to increase it.

> **Note**
>
> **High Priority:** Use the effective bandwidth of your computation as a metric when measuring performance and optimization benefits.

### [9.2.1. Theoretical Bandwidth Calculation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#theoretical-bandwidth-calculation)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#theoretical-bandwidth-calculation "Permalink to this headline")

Theoretical bandwidth can be calculated using hardware specifications available in the product literature. For example, the NVIDIA Tesla V100 uses HBM2 (double data rate) RAM with a memory clock rate of 877 MHz and a 4096-bit-wide memory interface.

Using these data items, the peak theoretical memory bandwidth of the NVIDIA Tesla V100 is 898 GB/s:

\(\(\left. \left( 0.877 \times 10^{9} \right. \times (4096/8) \times 2 \right) \div 10^{9} = 898\text{GB/s}\)\)

In this calculation, the memory clock rate is converted in to Hz, multiplied by the interface width (divided by 8, to convert bits to bytes) and multiplied by 2 due to the double data rate. Finally, this product is divided by 10<sup>9</sup> to convert the result to GB/s.

> **Note**
>
> Some calculations use 1024<sup>3</sup> instead of 10<sup>9</sup> for the final calculation. In such a case, the bandwidth would be 836.4 GiB/s. It is important to use the same divisor when calculating theoretical and effective bandwidth so that the comparison is valid.

> **Note**
>
> On GPUs with GDDR memory with ECC enabled the available DRAM is reduced by 6.25% to allow for the storage of ECC bits. Fetching ECC bits for each memory transaction also reduced the effective bandwidth by approximately 20% compared to the same GPU with ECC disabled, though the exact impact of ECC on bandwidth can be higher and depends on the memory access pattern. HBM2 memories, on the other hand, provide dedicated ECC resources, allowing overhead-free ECC protection.[^2]

### [9.2.2. Effective Bandwidth Calculation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#effective-bandwidth-calculation)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#effective-bandwidth-calculation "Permalink to this headline")

Effective bandwidth is calculated by timing specific program activities and by knowing how data is accessed by the program. To do so, use this equation:

\(\(\text{Effective\ bandwidth} = \left( {\left( B_{r} + B_{w} \right) \div 10^{9}} \right) \div \text{time}\)\)

Here, the effective bandwidth is in units of GB/s, B<sub>r</sub> is the number of bytes read per kernel, B<sub>w</sub> is the number of bytes written per kernel, and time is given in seconds.

For example, to compute the effective bandwidth of a 2048 x 2048 matrix copy, the following formula could be used:

\(\(\text{Effective\ bandwidth} = \left( {\left( 2048^{2} \times 4 \times 2 \right) \div 10^{9}} \right) \div \text{time}\)\)

The number of elements is multiplied by the size of each element (4 bytes for a float), multiplied by 2 (because of the read _and_ write), divided by 10<sup>9</sup> (or 1,024<sup>3</sup>) to obtain GB of memory transferred. This number is divided by the time in seconds to obtain GB/s.

### [9.2.3. Throughput Reported by Visual Profiler](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#throughput-reported-by-visual-profiler)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#throughput-reported-by-visual-profiler "Permalink to this headline")

For devices with [compute capability](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#cuda-compute-capability) of 2.0 or greater, the Visual Profiler can be used to collect several different memory throughput measures. The following throughput metrics can be displayed in the Details or Detail Graphs view:

- Requested Global Load Throughput
- Requested Global Store Throughput
- Global Load Throughput
- Global Store Throughput
- DRAM Read Throughput
- DRAM Write Throughput

The Requested Global Load Throughput and Requested Global Store Throughput values indicate the global memory throughput requested by the kernel and therefore correspond to the effective bandwidth obtained by the calculation shown under [Effective Bandwidth Calculation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#effective-bandwidth-calculation).

Because the minimum memory transaction size is larger than most word sizes, the actual memory throughput required for a kernel can include the transfer of data not used by the kernel. For global memory accesses, this actual throughput is reported by the Global Load Throughput and Global Store Throughput values.

It’s important to note that both numbers are useful. The actual memory throughput shows how close the code is to the hardware limit, and a comparison of the effective or requested bandwidth to the actual bandwidth presents a good estimate of how much bandwidth is wasted by suboptimal coalescing of memory accesses (see [Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory)). For global memory accesses, this comparison of requested memory bandwidth to actual memory bandwidth is reported by the Global Memory Load Efficiency and Global Memory Store Efficiency metrics.

****

  As an exception, scattered writes to HBM2 see some overhead from ECC but much less than the overhead with similar access patterns on ECC-protected GDDR5 memory.
