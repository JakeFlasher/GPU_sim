# LDGSTS burst probe

This CUDA probe produces a focused asynchronous-copy trace for SM86. Build the
workload with the local CUDA toolchain, capture it with `capture_nvbit.py`, and
replay the resulting trace with `sim-run`.

The current capture helper writes the dynamic protobuf, control sidecar,
command, environment, and a compact output summary. It does not perform a
campaign sweep or attach approval metadata.
