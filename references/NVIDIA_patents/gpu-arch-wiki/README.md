# GPU Architecture & Data Precision Formats

NVIDIA GPU 微架构演进与数据精度格式交互式知识库。

## Live Demo

[https://zznjupt.github.io/gpu-arch-wiki/web/](https://zznjupt.github.io/gpu-arch-wiki/web/)

## Content

- **NV 微架构演进** — Fermi 到 Rubin，SM 结构对比、架构图、参考文献
- **数据精度格式** — FP64/FP32/FP16/BF16/TF32/FP8/FP6/FP4 等格式详解与硬件支持矩阵

## Structure

```
user/           # Markdown 源文件 + 构建脚本
web/            # 前端页面（纯静态 HTML/CSS/JS）
references/     # 参考资料
```

## Build

```bash
python3 user/build_arch.py   # 从 markdown 生成 HTML
```
