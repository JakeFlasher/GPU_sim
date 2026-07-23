# Preserved binary: `rmsnorm_fwd_bf16_run`

The preserved binary is
`workloads/simbench_bw/l2/rmsnorm_fwd_bf16/rmsnorm_fwd_bf16_run` with SHA-256
`e174aa79447b2a77dc64b64936bd569ef22247f68535f0e765621cbe5e3d3240`.
It is not an entry in
`workloads/portable_lcd/preserved_binaries_manifest.json`: this is a
`simbench_bw` cell, not a `portable_lcd` cell.

The exact row in
`data/local_labels/campaign/simbench_build_witness_sm86.json` is
`rmsnorm_fwd_bf16`; it binds build-input-set SHA-256
`ca68e677256925163e8c6c11160e88d8cc582063b0de93aad6058361a0896c20`
and row-seal SHA-256
`cf0c96091ec9fde2e0c7c502290c5aa307d16949c29fa2a86175a46edc280df8`.

The same-path migration entries in
`tools/local_labels/witness_compat_registry.json` bind:

- `Makefile`: witness `1d01e7885f3fdc5b6cc36ee4700944d52646e0b4923d807b0587a22811de4232`
  to current pin `940f561bf77c2217e3190eae7fd74c1fec0a7d136c641d6518211e3b10ecf0c6`.
- `main.cu`: witness `6eefbf3fa8f725878065dd0f2cc4e9412bb95b0ef087b19434e64ca45eaa6d29`
  to current pin `2ccf2c0be1e88848f5e335ecc767bf214c08aafaa0438909b3e102c85b01848e`.

The durable-v2 cell at
`tools/local_labels/authorities/capture_roster_authority_sm86.durable.v2.json`
retains `roster_source=simbench_bw`, `resolution_status=runnable`, the binary
pin above, build-witness SHA-256
`d577b3850f5ed04c798b1475d06cd6f29e48f857c0dc5735f923157e26cc7750`,
and the row seal above. Do not rebuild or repin this binary while its
preserved-state contract is active.
