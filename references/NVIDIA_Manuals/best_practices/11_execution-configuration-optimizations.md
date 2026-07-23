---
title: "11. Execution Configuration Optimizations"
section: "11"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#execution-configuration-optimizations"
---

# [11. Execution Configuration Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#execution-configuration-optimizations)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#execution-configuration-optimizations "Permalink to this headline")

One of the keys to good performance is to keep the multiprocessors on the device as busy as possible. A device in which work is poorly balanced across the multiprocessors will deliver suboptimal performance. Hence, it’s important to design your application to use threads and blocks in a way that maximizes hardware utilization and to limit practices that impede the free distribution of work. A key concept in this effort is occupancy, which is explained in the following sections.

Hardware utilization can also be improved in some cases by designing your application so that multiple, independent kernels can execute at the same time. Multiple kernels executing at the same time is known as concurrent kernel execution. Concurrent kernel execution is described below.

Another important concept is the management of system resources allocated for a particular task. How to manage this resource utilization is discussed in the final sections of this chapter.
