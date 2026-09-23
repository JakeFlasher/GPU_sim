# Vortex SimX/RTL functional differential — 2026-09-23

The first independently developed open-GPU comparison completed for the
roadmap is a Vortex `vecadd` hostless program executed by both Vortex SimX and
Vortex RTL simulation. Both produced the same 16-element destination vector.

## Reproduction environment

- Vortex revision: `ab31c0c2f66977fceea05ab282c518bb26b6964b`
- Container: Ubuntu 22.04 image
  `sha256:b8b6ee6aa931ecd0d952abc34dc0e5f7c6a30c6bb71b079fe399fde0329c02`
- Vortex configuration: 1 cluster, 1 core, 4 warps, 4 threads, 2 memory banks
- Program: `tests/kernel/vecadd/vecadd.vxbin`
- Backends: `simx` and `rtlsim`

The host-side replay commands were:

```bash
export LD_LIBRARY_PATH=/tmp/vortex/third_party/ramulator
$VORTEX_BUILD/sim/simx/simx \
  $VORTEX_BUILD/tests/kernel/vecadd/vecadd.vxbin
$VORTEX_BUILD/sim/rtlsim/rtlsim \
  $VORTEX_BUILD/tests/kernel/vecadd/vecadd.vxbin
```

where `VORTEX_BUILD=/tmp/vortex/build`.

Both outputs were:

```text
0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75
```

The same vector is produced by the custom `simt_u32_v1` functional program
using explicit initial memory and load/add/store operations. The cross-model
check is reproducible with:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-arch --bin custom_simt
python tools/vortex_vecadd_cross_model.py \
  --custom-binary sim/target/release/custom_simt \
  --spec arch_profiles/specs/simt_u32_vecadd.json \
  --program arch_profiles/programs/simt_u32_vecadd.json \
  --simx-log docs/vortex_vecadd_simx_20260923.log \
  --rtlsim-log docs/vortex_vecadd_rtlsim_20260923.log \
  --output docs/vortex_vecadd_cross_model_20260923.json
```

The machine-readable result is
`docs/vortex_vecadd_simx_rtlsim_20260923.json`.

The custom-program cross-model result is
`docs/vortex_vecadd_cross_model_20260923.json`; it compares all 16 outputs
from the custom executor, SimX, and rtlsim, and all three vectors match.

## Boundary

This establishes an independent open-GPU functional oracle and a SimX/RTL
parity point. It does not yet adapt Vortex binaries into the custom executor,
compare cycle traces, or establish silicon accuracy. Vortex's RISC-V ISA and
runtime remain a separate adapter task; the structural `VX_config.toml`
adapter intentionally fails closed for execution until that work exists.
