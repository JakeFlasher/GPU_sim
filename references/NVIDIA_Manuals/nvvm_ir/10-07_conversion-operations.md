---
title: "10.7. Conversion Operations"
section: "10.7"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#conversion-operations"
---

## [10.7. Conversion Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#conversion-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#conversion-operations "Permalink to this headline")

Supported:

- `trunc .. to`
- `zext .. to`
- `sext .. to`
- `fptrunc .. to`
- `fpext .. to`
- `fptoui .. to`
- `fptosi .. to`
- `uitofp .. to`
- `sitofp .. to`
- `ptrtoint .. to`
- `inttoptr .. to`
- `addrspacecast .. to`
- `bitcast .. to`

See [Conversion](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#conversion) for a special use case of `bitcast`.
