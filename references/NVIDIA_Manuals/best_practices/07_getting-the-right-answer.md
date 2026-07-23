---
title: "7. Getting the Right Answer"
section: "7"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#getting-the-right-answer"
---

# [7. Getting the Right Answer](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#getting-the-right-answer)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#getting-the-right-answer "Permalink to this headline")

Obtaining the right answer is clearly the principal goal of all computation. On parallel systems, it is possible to run into difficulties not typically found in traditional serial-oriented programming. These include threading issues, unexpected values due to the way floating-point values are computed, and challenges arising from differences in the way CPU and GPU processors operate. This chapter examines issues that can affect the correctness of returned data and points to appropriate solutions.
