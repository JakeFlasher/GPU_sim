## Nvidia GPGPU 微架构演进

| uArch     | Year | Core  | SM(X/M) | GPC | TPC | INT32+FP32 | INT32/FP32 | INT32 | FP32 | FP64 | Warp. | Disp. | LD/ST | SFU | Tens.Core |
| -----     | ---- |-------| ------- | --- | --- | ---------- |------------|------ |------|------| ----- | ----- |-------|-----|-----------|
| Fermi     | 2010 | GF100 | 16      | 4   | -   | 32         | -          | -     | -    | -    | 2     | 2     | 16    | 4   | -         |
| Kepler    | 2012 | GK210 | 15      | -   | -   | 192        | -          | -     | -    | 64   | 4     | 8     | 32    | 32  | -         |
| Maxwell   | 2014 | GM204 | 16      | 4   | -   | 128        | -          | -     | -    | -    | 4     | 8     | 32    | 32  | -         |
| Pascal    | 2016 | GP100 | 60      | 6   | 5   | 64         | -          | -     | -    | 32   | 2     | 4     | 16    | 16  | -         |
| Volta     | 2017 | GV100 | 84      | 6   | 7   | -          | -          | 64    | 64   | 32   | 4     | 4     | 32    | 16  | 8 (1th)   |
| Turing    | 2018 | TU102 | 72      | 6   | 6   | -          | -          | 64    | 64   | 2    | 4     | 4     | 16    | 16  | 8 (2th)   |
| Ampere    | 2020 | GA100 | 128     | 8   | 8   | -          | -          | 64    | 64   | 32   | 4     | 4     | 32    | 16  | 4 (3th)   |
| Ampere    | 2020 | GA102 | 84      | 7   | 6   | -          | 64         | -     | 64   | 2    | 4     | 4     | 16    | 16  | 4 (3th)   |
| Ada       | 2022 | AD102 | 144     | 12  | 6   | -          | 64         | -     | 64   | 2    | 4     | 4     | 16    | 16  | 4 (4th)   |
| Hopper    | 2022 | GH100 | 144     | 8   | 9   | -          | -          | 64    | 128  | 64   | 4     | 4     | 32    | 16  | 4 (4th)   |
| Blackwell | 2024 | -     | -       | -   | -   | -          | -          | -     | -    | -    | -     | -     | -     | -   | 4 (5th)   |
| Rubin     | 2026 | -     | -       | -   | -   | -          | -          | -     | -    | -    | -     | -     | -     | -   | 4 (6th)   |

---

## Nvidia 微架构存储层次发展

| uArch     | Year | Core | Reg/File | Shared Mem | L1$ | L2$ | SMEM+L1 Config | Cache Strategy | Memory Interface |
| -----     | ---- | ---- | -------- | ---------- | --- | --- | -------------- | -------------- | ---------------- |
| Tesla     | 2006 | G80    | 8KB      | 16KB       | -   | -   | -              | -              | GDDR3            |
| Fermi     | 2010 | GF100  | 32KB     | 48KB       | 16KB| 768KB| Configurable  | Write-Back     | GDDR5            |
| Kepler    | 2012 | GK110  | 64KB     | 64KB       | 64KB| 1.5MB| Configurable  | Write-Back     | GDDR5            |
| Maxwell   | 2014 | GM204  | 64KB     | 96KB       | 32KB| 2MB  | Fixed Split    | Write-Through  | GDDR5            |
| Pascal    | 2016 | GP102  | 64KB     | 64/96KB    | 32KB| 2MB  | Fixed Split    | Write-Through  | GDDR5X           |
| Volta     | 2017 | GV100  | 64KB     | 96KB       | 128KB| 6MB | Fixed Split    | Write-Through  | HBM2             |
| Turing    | 2018 | TU102  | 64KB     | 64KB       | 4KB | 5.5MB| Fixed Split    | Write-Through  | GDDR6            |
| Ampere    | 2020 | GA102  | 64KB     | 164KB      | -   | 40MB | 128KB Shared   | -              | GDDR6X/HBM2e     |
| Ada       | 2022 | AD102  | 64KB     | 99KB       | -   | 50MB | 128KB Shared   | -              | GDDR6X           |
| Hopper    | 2022 | GH100  | 64KB     | 256KB      | -   | 50MB | 256KB Shared   | -              | HBM3             |
| Blackwell | 2024 | GB200  | 64KB     | 228KB      | -   | 140MB| 228KB Shared   | -              | HBM3e            |
| Rubin     | 2026 | TBD    | TBD      | TBD        | -   | TBD  | TBD            | -              | HBM4             |

> **注：** Shared Memory / L1 Cache 大小在不同架构中有不同的配置策略。存储层次结构说明：<br/>
> - **Register File (寄存器文件)**: 每个 SM 内部的高速存储，供线程使用<br/>
> - **Shared Memory (共享内存)**: 每个 SM 内部，供该 SM 内所有线程块共享<br/>
> - **L1$ (L1缓存)**: 每个 SM 级别的缓存<br/>
> - **L2$ (L2缓存)**: 芯片全局缓存，所有 SM 共享<br/>
> - **Memory Interface (内存接口)**: 连接到外部 GDDR/HBM 内存

---

---
id: arch-fermi
name: Fermi
year: 2010
tags: 首个完整 GPU 计算架构
image: fermi_sm.png
---

### SM 配置
- Warp Scheduler: 2
- Dispatch Unit: 2
- INT32+FP32 Core: 32
- LD/ST Unit: 16
- SFU: 4

### 说明
- Fermi 是 NVIDIA 第一个面向通用计算设计的完整架构。每个 SM 包含 32 个 CUDA Core，分布在 2 条 lane 上（每条 16 个）。
- Fermi 架构把 Tesla 架构的 TPC 改名为 **Graphics Processor Clusters（GPC）**，毕竟 Texture 现在显得不再那么重要。
- Fermi 的 CUDA core 实现了浮点乘加融合（FMA），每个 CUDA Core 内部是 **1 个单精度浮点单元 (FPU) + 1 个整数单元 (ALU)**，可以直接执行 FMA 操作。每个 cycle 可跑 16 个双精度 FMA。Tesla 的浮点并没有完全按照 IEEE754 标准实现，例如不支持 subnormal 浮点，而 Fermi 实现了完整的支持，并且实现了 IEEE754 标准的 rounding mode。
- Fermi 架构把 **Load/Store（LD/ST）**单元独立出来，地址空间也从 32 位扩大到了 64 位。寄存器堆保存了 32768 个 32 位寄存器。


### 规模+存储层次
- Fermi 架构引入了 L1 和 L2 数据缓存。
- Fermi 架构的 Shared Memory 和 L1 数据缓存大小是可配置的，二者共享 64 KB 的空间，可以选择 48KB Shared Memory 加 16KB 的 L1 数据缓存，也可以选择 16KB Shared Memory 和 48KB 的 L1 数据缓存。
- Fermi 架构的 L2 缓存采用的是 **写回（write-back）** 策略。

### References
1. [Fermi: NVIDIA’s Next Generation CUDA Compute Architecture](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf)
2. [NVIDIA’s Fermi: The First Complete GPU Computing Architecture](https://www.nvidia.com/content/PDF/fermi_white_papers/P.Glaskowsky_NVIDIA's_Fermi-The_First_Complete_GPU_Architecture.pdf)
3. [NV微架构演进](https://gcne582ug2ks.feishu.cn/wiki/IiCfwVzWKi7JmtkTOI5ciFcknxd)

---

---
id: arch-kepler
name: Kepler
year: 2012
tags: SMX, FP64
image: gk210_sm.png
---

### SM 配置
- Warp Scheduler: 4
- Dispatch Unit: 8
- INT32+FP32 Core: 192
- FP64 Unit: 64
- LD/ST Unit: 32
- SFU: 32

### 说明
- Kepler 去掉了 TPC/GPC 这一个层级，而是把 SM 做的很大，称为 **SMX**。相比 Fermi 大幅堆料：CUDA Core 从 32 暴增到 192（4 × 3 × 16，每条 lane 仍是 16 个）。
- Kepler 引入 **独立的 64 个双精度运算单元**，不需要通过单精度单元去做双精度运算。这使得 Kepler 的双精度性能远超前后几代。4 个 Warp Scheduler 搭配 8 个 Dispatch Unit（1:2 比例）。
- Kepler 相比 Fermi 架构的主要改进：
  - Dynamic Parallelism 和 Grid Management Unit：不仅 CPU 可以提交任务到 GPU 执行，GPU 自己也可以提交任务到自己上去执行
  - Hyper-Q：允许多个 CPU 核心同时向 GPU 提交任务，把硬件任务队列从 1 增加到了 32 个。每个 CUDA stream 会对应到一个硬件任务队列，因此增加硬件任务队列，可以减少 false dependency。
  - GPUDirect：支持 RDMA

### 规模+存储层次 (GK210)
[full-image: gk210_full.png]

- Kepler 引入了一个额外的 48KB 只读 Data Cache，用于保存只读的数据，可以提供相比 Shared/L1 cache 更高的性能。
- 根据 [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf)，Kepler 架构每个周期每个 SM 可以读取 256 字节的数据，也就是说，每个 LD/ST unit 每周期可以读取 4 字节的数据。

### References
- [NVIDIA’s Next Generation CUDA Compute Architecture: Kepler TM GK110/210](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf)
- [NVIDIA® KEPLER GK110 NEXT-GENERATION CUDA® COMPUTE ARCHITECTURE](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/documents/NV-DS-Tesla-KCompute-Arch-May-2012-LR.pdf)


---

---
id: arch-maxwell
name: Maxwell
year: 2014
tags: SMM, 能效比优先
image: maxwell_sm.png
---

### SM 配置
- Warp Scheduler: 4
- Dispatch Unit: 8
- INT32+FP32 Core: 128
- LD/ST Unit: 32
- SFU: 32

### 说明
- 虽然 Kepler 把 GPC 层次去掉了，但是 Maxwell 架构又把 GPC 加回来了。
- SM 更名为 **SMM**。开始做减法：**移除独立双精度计算单元**，CUDA Core 从 192 精简到 128（4 × 32）。
- Maxwell 把计算单元也划分成了四份，**每一份叫做一个 Processing Block（PB）**，每个 Block 有独立的 Warp Scheduler、Instruction Buffer 和 Register File。
- 工艺提升带来的收益：每个 CUDA Core 性能比 Kepler 提升 **1.4x**，整体能效比提升 **2x**。

### 规模+存储层次 (gm204，最大规模芯片 gm200 无官方文档)
[full-image: gm204_full.png]

- Maxwell 架构的 L1 缓存和 Shared Memory 不再共享，Shared Memory 独占 96KB，然后 L1 缓存和 Texture 缓存共享空间。
- 根据 [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf)，Maxwell 架构每个周期每个 SM 可以读取 256 字节的数据，也就是说，每个 LD/ST unit 每周期可以读取 4 字节的数据。

### References
- [NVIDIA GeForce GTX 980 Whitepaper](https://www.microway.com/download/whitepaper/NVIDIA_Maxwell_GM204_Architecture_Whitepaper.pdf)
- [New GPU Features of NVIDIA's Maxwell Architecture](https://developer.download.nvidia.cn/assets/events/GDC15/GEFORCE/Maxwell_Archictecture_GDC15.pdf)
- [5 Things You Should Know About the New Maxwell GPU Architecture](https://developer.nvidia.com/blog/5-things-you-should-know-about-new-maxwell-gpu-architecture/)

---

---
id: arch-pascal
name: Pascal
year: 2016
tags: FP16, NVLink
image: gp100_sm.png
---

### SM 配置
- Warp Scheduler: 2
- Dispatch Unit: 4
- INT32+FP32 Core: 64
- FP64 Unit: 32
- LD/ST Unit: 16
- SFU: 16

### 说明
- Pascal 架构每个 SM 只有 2 个 PB。
- **NVLink**：首次引入 GPU 间高速互联 NVLink 1.0，双向带宽 160 GB/s
- SM 内部进一步精简：2 × 32 = 64 CUDA Core，Warp Scheduler 也缩减为 2 个。
- **双精度回归**：32 个独立 FP64 单元（2 × 16）
- **FP16 支持**：CUDA Core 首次支持半精度计算，为后续 AI 加速铺路
- 支持 Compute Preemption，使得 kernel 可以在指令级别做抢占，而不是 thread block 级别，这样就可以让调试器等交互式的任务不会阻碍其他计算任务的进行；在 Kepler 架构中，只有等一个 thread block 的所有 thread 完成，硬件才可以做上下文切换，但是如果中间遇到了调试器的断点，这时候 thread block 并没有完成，那么此时只有调试器可以使用 GPU，其他任务就无法在 GPGPU 上执行

### 规模+存储层次 (GP100)
[full-image: gp100_full.png]

- 支持 Unified Memory，使得 CPU 和 GPU 可以共享虚拟地址空间，让数据自动进行迁移。
- 8 个 512 位的内存控制器，每个内存控制器附带 512 KB L2 缓存，总共有 4096 KB 的 L2 缓存。
- 每两个内存控制器为一组，连接到 4 个 1024 位的 HBM2 内存。
- Pascal 架构每个 SM 有 64 KB 的 Shared memory，并且 SM 的数量比 Maxwell 的两倍还要多，因此实际上是在变相地增加 Shared memory 的数量、容量以及带宽。
- 根据 [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf)，Pascal 架构每个周期每个 SM 可以读取 128 字节的数据，也就是说，每个 LD/ST unit 每周期可以读取 8 字节的数据。

### SM 微架构图
[full-image: pascal_sm_arch.png]

### References
- [NVIDIA Tesla P100 Whitepaper](https://images.nvidia.cn/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)

---

---
id: arch-volta
name: Volta
year: 2017
tags: V100, Tensor Core 诞生
image: gv100_sm.png
---

### SM 配置
- Warp Scheduler: 4
- Dispatch Unit: 4
- FP32 Core: 64
- INT32 Core: 64
- FP64 Core: 32
- Tensor Core: 8 (1st) [highlight]
- LD/ST Unit: 32
- SFU: 16

### 说明
- **里程碑架构。**
- **Tensor Core 1.0**：每个 SM 8 个（4 × 2），每个 TC 每周期执行 4×4×4 GEMM（FP16 输入，FP32 累加），等效 64 个 FP32 ALU。
- **NVLink 2.0**
- **CUDA Core 拆分**：不再是统一的 FPU+ALU，而是独立的 FP32 Core 和 INT32 Core。优势：**两者可以同时执行**，混合运算吞吐翻倍。
- **独立线程调度**：每个线程拥有独立 PC 和调用栈
- Warp Scheduler 变为 1:1 对应 Dispatch Unit（之前是 1:2），并增加 L0 ICache。
- 每个 SM 拆分成 4 个 PB 回归。
- Volta 的 Warp Scheduler 又回到了单发射，一条涉及 32 条线程的指令被发射，那么它需要两个周期来完成，第二个周期的时候，Warp Scheduler 也会同时发射其他指令，从而实现指令级并行。

### 规模+存储层次 (GV100)
[full-image: gv100_full.png]

- 在 Volta 架构中，L1 Data Cache 和 Shared memory 再次共享。
- 引入了 L0 Instruction Cache，每个 Processing Block 内部都有一个。
- 8 个 512 位的内存控制器。
- 根据 [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf)，Volta 架构每个周期每个 SM 可以读取 256 字节的数据，也就是说，每个 LD/ST unit 每周期可以读取8 字节的数据。
- 根据 [gpu-benches](https://github.com/te42kyfo/gpu-benches) 实测，每个 SM 每周期只能读取不到 128 字节（14 TB/s，80 个 SM，时钟频率 1530 MHz，每个 SM 每周期读取 114 字节）的数据（V100不是完整GV100核心，拥有 80 个 SM）。
- 6144 KB 的 L2 缓存
  - 分为 64 个 L2 slice，每个 slice 是 96 KB 的大小。
  - 每个 slice 每周期可以读取 32 B 的数据，因此整个 L2 缓存的读带宽是 2048 字节每周期。
  - L2 缓存工作在和 SM 同一个频率下，按 1530 MHz 频率来算，L2 缓存带宽是 3.133 TB/s，V100 的内存带宽是 0.9 TB/s，每个 SM 每个周期可以分到的 L2 带宽是 25.6 字节。


### SIMT 改进：独立线程调度
[images: simt_old.png | simt_volta.png]
[captions: Pascal 及之前：分支两侧串行执行 | Volta：独立 PC + __syncwarp()，分支可交错调度]

Pascal 及之前，一个 Warp 遇到分支时两侧只能串行执行。Volta 将 PC 和调用栈设计为每个线程独立，分支内的指令可以在更细的粒度上混合调度，并支持 `__syncwarp()` 显式同步。

### References
- [Tuning CUDA Applications for Volta](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html)
- [VOLTA: PROGRAMMABILITY AND PERFORMANCE](https://old.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.132-Volta-Choquette-NVIDIA-Final3.pdf)

---

---
id: arch-turing
name: Turing
year: 2018
tags: RTX 20 系列, RT Core
image: tu102_sm.png
---

### SM 配置
- Warp Scheduler: 4
- Dispatch Unit: 4
- FP32 Core: 64
- INT32 Core: 64
- Tensor Core: 8 (2nd) [highlight]
- LD/ST Unit: 16
- SFU: 16

### 说明
- **Tensor Core 2.0**：在 Volta 基础上**新增 INT8 和 INT4 支持**，开始覆盖推理量化场景。
- 引入 **RT Core**（光线追踪加速单元）。
- SM 内计算单元与 Volta 类似，每个 SM 含 4 个 Processing Block。

### 规模+存储层次 (TU102)
[full-image: tu102_full.png]

- 12 个 32-bit GDDR6 memory controller
- Turing 架构的每 TPC 的 L1 带宽是 Pascal 架构的两倍。


### SM 微架构图
[full-image: turing_sm_arch.png]

### References
- [NVIDIA TURING GPU ARCHITECTURE](https://images.nvidia.cn/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)
- [RTX ON – THE NVIDIA TURING GPU](https://old.hotchips.org/hc31/HC31_2.12_NVIDIA_final.pdf)

---

---
id: arch-ampere-ga100
name: Ampere (GA100)
year: 2020
tags: A100, 稀疏 Tensor Core
image: ga100_sm.png
---

### SM 配置
- Warp Scheduler: 4
- Dispatch Unit: 4
- FP32 Core: 64
- INT32 Core: 64
- FP64 Core: 32
- Tensor Core: 4 (3rd) [highlight]
- LD/ST Unit: 32
- SFU: 16

### 说明
- 4 个 PB。
- **Tensor Core 3.0** 
  - 数量从 8 减半到 4，但**每个吞吐量翻 4 倍**，总吞吐翻倍。
  - 数据类型大幅扩展：FP16、BF16、TF32、FP64、INT8、INT4、Binary。
  - 支持结构性稀疏矩阵。
- **NVLink 3.0**
- **‌Multi-Instance GPU**‌（多实例 GPU，**MIG 1.0**）

### 规模+存储层次

[full-image: ga100_full.png]

- 6 个 HBM2 stack，对应 12 个 512-bit memory controller。
- 每个 SM 的 L1 Data Cache/Shared Memory 总量增加到了 192 KB。
- A100 GPU 有 40 MB 的 L2 缓存，分为两个 partition。
- 每个 partition 有 40 个 L2 slice，每个 slice 是 512 KB 的大小。
- 每 8 个 L2 slice 对应一个 memory controller
- 每个 slice 每周期可以读取 64B 的数据，因此整个 L2 缓存的读带宽是 5120 字节每周期
- L2 缓存工作在和 SM 同一个频率下，按 1410 MHz 频率来算，L2 缓存带宽是 7.219 TB/s，A100 的内存带宽是 1.555 TB/s，
- 每个 SM 每个周期可以分到的 L2 带宽是 47.4 字节
- 根据 [gpu-benches](https://github.com/te42kyfo/gpu-benches) 实测，每个 SM 每周期只能读取不到 128 字节（19 TB/s，108 个 SM，时钟频率 1410 MHz，每个 SM 每周期读取 125 字节）的数据。
- A100 GPU 有 108 个 SM（不是完整的GA100），一共 432 个 Tensor Core，每个 Tensor Core 每周期可以进行 256 个 FP16 FMA 计算，SM 频率 1410 MHz，因此 A100 的 FP16 Tensor Core 峰值性能是 432 * 256 FLOPS * 2 * 1410 MHz = 312 TFLOPS。

### A100 vs V100 性能提升
[full-image: a100_vs_v100.png]

### References
- [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA A100 GPU: PERFORMANCE & INNOVATION FOR GPU COMPUTING](https://hc32.hotchips.org/assets/program/conference/day1/HotChips2020_GPU_NVIDIA_Choquette_v01.pdf)
- [The A100 Datacenter GPU and Ampere Architecture](https://ieeexplore.ieee.org/abstract/document/9365803)
- [Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis](https://ieeexplore.ieee.org/abstract/document/9926299/)

---

---
id: arch-ampere-ga102
name: Ampere (GA102)
year: 2020
tags: RTX 30
image: ga102_sm.png
---

### SM 配置
- Warp Scheduler: 4
- Dispatch Unit: 4
- FP32 Core: 64
- INT32/FP32 Core: 64
- Tensor Core: 4 (3rd) [highlight]
- LD/ST Unit: 16
- SFU: 16

### 说明
- 4 个 PB。
- 出现了 FP32/INT32 混合的 core，使得 FP32 峰值性能翻倍，但是这个峰值也更难达到，因为达到峰值意味着不用到 FP32/INT32 core 的 INT32 部分。

### 规模+存储层次

[full-image: ga102_full.png]

- 12 个 32-bit memory controller，一共 384 位。
- 12 组 512KB 的 L2 缓存，每组对应一个内存控制器，L2 一共是 6144 KB。
- GA102 的 shared memory 带宽是每个 SM 每个时钟 128 字节，而 Turing 架构的这个值是 64。
- GeForce RTX 3080 (GA102) 的每 SM L1 带宽是 219 GB/s（一个 SM 有 16 个 LD/ST unit，每个 LD/ST unit 每个周期读取 8B 的数据，所以带宽是 219 GB/s）。
- GeForce RTX 2080 Super (TU104) 的每 SM L1 带宽是 116 GB/s（每个 SM 有 16 个 LD/ST unit，每个 LD/ST unit 每个周期读取 4B 的数据，带宽是 116 GB/s）。


### SM 微架构图
[full-image: 1-GA102_sm.png]

### References
- [NVIDIA AMPERE GA102 GPU ARCHITECTURE](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)

---

---
id: arch-ada
name: Ada Lovelace
year: 2022
tags: RTX 40 系列, FP8 / DLSS 3
image: ad102_sm.png
---

### SM 配置
- Warp Scheduler: 4
- Dispatch Unit: 4
- FP32 Core: 64
- INT32/FP32 Core: 64
- Tensor Core: 4 (4th) [highlight]
- LD/ST Unit: 16
- SFU: 16

### 说明
**消费级旗舰。**延续 Ampere 的双数据通路（64 专用 FP32 + 64 Flex FP32/INT32），合计 128 FP32 Core。TSMC 4N 工艺。
- **Tensor Core 4.0：**新增 FP8（E4M3 / E5M2）支持，搭载 Hopper 的 Transformer Engine 技术。吞吐量相比 Ampere 再翻倍。
- **第 3 代 RT Core：**新增 Opacity Micro-Map（OMM）和 Displaced Micro-Mesh Engine，光追性能 2x。
- 支持 DLSS 3（帧生成）、Shader Execution Reordering（SER，光追调度优化）。
- **AD102**：144 SM。
- **RTX 4090**：128 SM。

### 规模+存储层次 (AD102)

[full-image: ad102-full.png]

- 12 个 32-bit memory controller，一共 384 位。
- **L2 缓存大幅扩容：**，AD102 达 96MB（Ampere GA102 仅 6MB，16x 增长），显著减少显存访问。

### References
- [NVIDIA ADA GPU ARCHITECTURE](https://images.nvidia.cn/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf)

---

---
id: arch-hopper
name: Hopper
year: 2022
tags: H100 / H200, Transformer Engine
image: gh100_sm.png
---

### SM 配置
- Warp Scheduler: 4
- Dispatch Unit: 4
- FP32 Core: 128
- INT32 Core: 64
- FP64 Core: 64
- Tensor Core: 4 (4th) [highlight]
- LD/ST Unit: 32
- SFU: 16

### 说明
**大模型时代的数据中心旗舰。**
- 8 个 GPC，66 个 TPC，每个 TPC 有两个 SM；一共 132 个 SM。（H100 SXM5 参数）
- **Tensor Core 4.0：**
  - 原生 FP8 支持（E4M3 / E5M2）。
  - 引入 `wgmma`（Warp Group MMA）指令，128 线程协作完成矩阵运算。
  - FP16 吞吐 729 TFLOPS。
  - FP8 达 1,448 TFLOPS（H800）。
- **Transformer Engine：**硬件动态精度选择，每层自动在 FP8 和 FP16 之间切换，训练精度不损失。
- **Thread Block Cluster：**跨 SM 的线程块协作原语。
- **Distributed Shared Memory (DSMEM)：**SM 间可直接访问彼此的 Shared Memory，延迟 180 cycles，带宽 3.27 TB/s。
- **Tensor Memory Accelerator (TMA)：**异步 1D-5D 张量搬运，解放计算线程。
- **DPX 指令集：**动态规划算法加速（Smith-Waterman 等），最高 13x 提速。
- **H100 SXM5**：132 SM，16,896 FP32 Core，528 TC，80GB HBM3（3 TB/s），50MB L2，800 亿晶体管，TSMC N4。
- **NVLink 4.0**
- **MIG 2.0**

### 规模+存储层次 (GH100)

[full-image: gh100-full.png]

- Shared Memory + L1 扩大至 256KB。
- HBM3 DRAM，5 个 stack，10 个 512-bit memory controller，总共 80 GB 容量。（H100 SXM5 参数）
- 完整版的 GH100 芯片有 60MB 的 L2 缓存，H100 有 50MB 的 L2 缓存。
- 根据 [gpu-benches](https://github.com/te42kyfo/gpu-benches) 实测，每个 SM 每周期只能读取略多于 128 字节（25 TB/s，114 个 SM，时钟频率 1620 MHz，每个 SM 每周期读取 135 字节）的数据。

### CUDA Kernel
CUDA Kernel 之前是三个层次：Grid、Thread Block 和 Thread，分别对应整个 GPU、SM 和 CUDA Core，而这一代引入了 Thread Block Cluster 的层次，变成了四个层次：Grid、Thread Block Cluster、Thread Block 和 Thread。其中 Thread Block 对应 GPC，每个 GPC 有多个 TPC，每个 TPC 有多个 SM。

### References
- [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core)
- [NVIDIA HOPPER GPU: SCALING PERFORMANCE](https://hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.NVIDIA.Choquette.vfinal01.pdf)

---

---
id: arch-blackwell
name: Blackwell
year: 2024
tags: B100 / B200, FP4 / TMEM
image: gb202_sm.png
---

### SM 配置
- Tensor Core: 4 (5th) [highlight]
- TMEM (新增): 256KB
- SMEM + L1: 228KB

### 说明
**双芯片封装，2080 亿晶体管。**INT32 和 FP32 执行路径统一，INT32 Core 数量与 FP32 持平。

- **Tensor Core 5.0：**
  - 全新 FP4（e2m1）和 FP6（e3m2 / e2m3）支持。
  - FP4 吞吐 7,702 TFLOPS。
  - FP8 吞吐 3,851 TFLOPS（B200）。
  - 引入 `tcgen05.mma` 单线程 tensor 指令，替代 warp 级 MMA，延迟从 32 cycles 降至 11 cycles。
- **Tensor Memory (TMEM)：**全新的 256KB 专用 Tensor Core 存储，512 列 × 128 lane 的 2D 阵列。读带宽 16 TB/s，写带宽 8 TB/s（每 SM）。缓存未命中延迟降低 58%。
- **硬件解压引擎：**原生支持 LZ4、Snappy、Zstd、GZIP 解压。
- **B200**：148 SM，192GB HBM3e。**GB202**：192 SM，24,576 FP32 Core，TSMC N4P。
- **LLM 性能：**Mistral-7B FP8 推理 78,400 tok/s（1.59x vs H200），FP4 达 112,800 tok/s。

---

---
id: arch-rubin
name: Rubin
year: 2026 (预计)
tags: R100, HBM4
---

### SM 配置
- Tensor Core: 4 (6th) [highlight]

### 说明
- **下一代数据中心平台，预计 2026 下半年。**
- **Tensor Core 6.0：**支持 FP4/FP6/FP8/FP16/BF16/TF32/FP32/FP64。
- **第 3 代 Transformer Engine**：硬件自适应压缩 + 跨层动态精度选择 + 双级微块缩放（NVFP4）。
- **HBM4 显存：**288GB，8 个堆叠，带宽 22 TB/s（Blackwell 的 2.8 倍）。
- **NVLink 6.0：**每 GPU 双向 3.6 TB/s（NVLink 5.0 的 2 倍）。
- **FP4 算力：**50 PFLOPS（Blackwell 20 PFLOPS 的 2.5 倍）。
- **Vera Rubin NVL72 系统：**72 GPU 聚合 FP4 推理 3,600 PFLOPS，NVLink 聚合带宽 260 TB/s。

