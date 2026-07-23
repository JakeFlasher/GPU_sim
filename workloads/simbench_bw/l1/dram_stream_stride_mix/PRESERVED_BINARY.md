# Preserved binary: `dram_stream_stride_mix_run`

The preserved binary is
`workloads/simbench_bw/l1/dram_stream_stride_mix/dram_stream_stride_mix_run`
with SHA-256
`51d4f86a01a63c89e537d578e580fca3420f89c9c3b1eeda63cb9cf8e7c48040`.
It is not an entry in
`workloads/portable_lcd/preserved_binaries_manifest.json`: this is a
`simbench_bw` cell, not a `portable_lcd` cell.

The exact row in
`data/local_labels/campaign/simbench_build_witness_sm86.json` is
`dram_stream_stride_mix`; it binds build-input-set SHA-256
`976653f98777758bb241c32e99657ffd255fa247262678ef2ef2639e8f7cda94`
and row-seal SHA-256
`99e603e6b984cb266e836818b4033a89f435c2faaaa188ec7a8220036531c0d4`.

The same-path migration entries in
`tools/local_labels/witness_compat_registry.json` bind:

- `Makefile`: witness `a3a9e8436c1a5b0559d59f50e795eef83311ddfc47acfe04b538e73e29fd12db`
  to current pin `c5460f516ab4f648528c0e0d787e3aa29341e350614ac167d5e40932495df04d`.
- `main.cu`: witness `47d33b81a89717f35f9afeff69477c2dc2bc8fb01f68011b47a2961f5743703d`
  to current pin `815ba11c22205c11603e4a53dcbe662d38b6137b5264176ea19946a653088b28`.

The durable-v2 cell at
`tools/local_labels/authorities/capture_roster_authority_sm86.durable.v2.json`
retains `roster_source=simbench_bw`, `resolution_status=runnable`, the binary
pin above, build-witness SHA-256
`d577b3850f5ed04c798b1475d06cd6f29e48f857c0dc5735f923157e26cc7750`,
and the row seal above. Do not rebuild or repin this binary while its
preserved-state contract is active.
